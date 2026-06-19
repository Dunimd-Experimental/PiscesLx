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

use serde::{Deserialize, Serialize};

/// Directories to skip entirely during codebase search.
const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "build", "dist", "venv",
    ".venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".eggs", ".svn", ".hg",
];

fn should_skip_dir(name: &str) -> bool {
    name.starts_with('.') || SKIP_DIRS.contains(&name)
}

fn is_binary(content: &[u8]) -> bool {
    content[..content.len().min(8192)].contains(&0x00)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchResult {
    pub file_path: String,
    pub line_number: usize,
    pub line_content: String,
    pub score: f64,
}

pub fn search_codebase(query: &str, path: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let walker = walkdir::WalkDir::new(path)
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() {
                return e
                    .file_name()
                    .to_str()
                    .map(|s| !should_skip_dir(s))
                    .unwrap_or(false);
            }
            // Skip dot-files
            if e.file_name().to_str().map_or(false, |s| s.starts_with('.')) {
                return false;
            }
            true
        });

    let query_lower = query.to_lowercase();
    let terms: Vec<&str> = query_lower.split_whitespace().collect();
    if terms.is_empty() {
        return results;
    }

    for entry in walker.filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }

        // Check file size before reading
        if let Ok(meta) = entry.metadata() {
            if meta.len() > 2 * 1024 * 1024 {
                continue; // skip files > 2 MB
            }
        }

        let content = match std::fs::read(entry.path()) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Skip binary files
        if is_binary(&content) {
            continue;
        }

        let content_str = match String::from_utf8(content) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let file_path = entry.path().to_string_lossy().to_string();
        for (i, line) in content_str.lines().enumerate() {
            let line_lower = line.to_lowercase();
            let mut score = 0.0_f64;
            for term in &terms {
                if line_lower.contains(term) {
                    score += 1.0;
                }
            }
            if score > 0.0 {
                results.push(SearchResult {
                    file_path: file_path.clone(),
                    line_number: i + 1,
                    line_content: line.to_string(),
                    score,
                });
            }
        }
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(50);
    results
}

pub fn grep(
    pattern: &str,
    path: &str,
    case_insensitive: bool,
    glob_filter: Option<&str>,
) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let re = if case_insensitive {
        regex::RegexBuilder::new(pattern)
            .case_insensitive(true)
            .build()
            .ok()
    } else {
        regex::Regex::new(pattern).ok()
    };

    let re = match re {
        Some(r) => r,
        None => return results,
    };

    let glob_matcher = glob_filter.and_then(|g| glob::Pattern::new(g).ok());

    let walker = walkdir::WalkDir::new(path)
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() {
                return e
                    .file_name()
                    .to_str()
                    .map(|s| !should_skip_dir(s))
                    .unwrap_or(false);
            }
            if e.file_name().to_str().map_or(false, |s| s.starts_with('.')) {
                return false;
            }
            true
        });

    for entry in walker.filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        if let Some(ref matcher) = glob_matcher {
            let fname = entry.file_name().to_string_lossy();
            if !matcher.matches(&fname) {
                continue;
            }
        }

        let content = match std::fs::read(entry.path()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if is_binary(&content) {
            continue;
        }
        let content_str = match String::from_utf8(content) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let file_path = entry.path().to_string_lossy().to_string();
        for (i, line) in content_str.lines().enumerate() {
            if re.is_match(line) {
                results.push(SearchResult {
                    file_path: file_path.clone(),
                    line_number: i + 1,
                    line_content: line.to_string(),
                    score: 1.0,
                });
            }
        }
    }

    results
}

pub fn glob(pattern: &str, path: &str) -> Vec<String> {
    let mut results = Vec::new();
    let walker = walkdir::WalkDir::new(path).follow_links(true).into_iter();
    let glob_pattern = match glob::Pattern::new(pattern) {
        Ok(p) => p,
        Err(_) => return results,
    };

    for entry in walker.filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path().to_string_lossy().to_string();
        if glob_pattern.matches(&p) {
            results.push(p);
        }
    }

    results

}
