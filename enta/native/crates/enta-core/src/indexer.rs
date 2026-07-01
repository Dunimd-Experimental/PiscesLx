//! Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
//!
//! This file is part of PiscesL1.
//! The PiscesL1 project belongs to the Dunimd Team.
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

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Stop words — English words unlikely to be meaningful in code search
// ---------------------------------------------------------------------------

const STOP_WORDS: &[&str] = &[
    "the", "this", "that", "with", "from", "have", "been", "were",
    "they", "their", "will", "would", "could", "should", "about",
    "which", "when", "where", "what", "into", "over", "after",
    "before", "between", "under", "above", "there", "here",
    "also", "than", "then", "just", "only", "very", "much",
    "such", "each", "every", "both", "some", "these", "those",
    "because", "while", "during", "through", "other", "being",
];

fn is_stop_word(s: &str) -> bool {
    STOP_WORDS.binary_search(&s).is_ok()
}

// ---------------------------------------------------------------------------
// Tokenizer — code-aware with camelCase / snake_case splitting
// ---------------------------------------------------------------------------

/// Tokenise source code text into searchable terms.
///
/// Preserves original case for camelCase detection, then lowercases
/// each sub-token. Handles:
/// - snake_case → `snake` + `case`
/// - camelCase → `camel` + `case`
/// - UPPERFirst → `upper` + `first`  (consecutive caps)
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = Vec::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            // Must start with letter or underscore
            if chars.is_empty() && !ch.is_ascii_alphabetic() && ch != '_' {
                continue;
            }
            chars.push(ch);
        } else {
            // End of identifier — split and emit
            split_identifier(&chars, &mut tokens);
            chars.clear();
        }
    }
    split_identifier(&chars, &mut tokens);
    tokens
}

/// Split a single identifier into sub-tokens, lowercased.
///
/// Boundaries (detected on ORIGINAL case before lowering):
/// 1. underscore → split and skip the underscore
/// 2. camelCase: lowercase→uppercase  (e.g. `renderHTML` → `render` + `html`)
/// 3. UPPER→lower within sequence: last cap starts new token (e.g. `HTMLElement` → `html` + `element`)
fn split_identifier(chars: &[char], tokens: &mut Vec<String>) {
    if chars.len() <= 1 {
        return;
    }
    let mut start = 0;
    let len = chars.len();

    for i in 1..len {
        // 1. underscore boundary (skip the underscore itself)
        if chars[i] == '_' {
            emit_segment(&chars[start..i], tokens);
            start = i + 1;
            continue;
        }
        // 2. camelCase: xY  (lowercase → uppercase)
        if chars[i - 1].is_ascii_lowercase() && chars[i].is_ascii_uppercase() {
            emit_segment(&chars[start..i], tokens);
            start = i;
            continue;
        }
        // 3. consecutive caps boundary: XYz → X-Yz
        //    `HTMLElement`: H T M L → H T M are caps, L→e is lower
        //    split so last cap (L) starts the new token
        if i > 1
            && chars[i - 2].is_ascii_uppercase()
            && chars[i - 1].is_ascii_uppercase()
            && chars[i].is_ascii_lowercase()
        {
            emit_segment(&chars[start..i - 1], tokens);
            start = i - 1;
        }
    }

    if start < len {
        emit_segment(&chars[start..], tokens);
    }
}

fn emit_segment(seg: &[char], tokens: &mut Vec<String>) {
    // Strip leading/trailing underscores
    let trimmed: String = seg.iter().filter(|&&c| c != '_').collect();
    if trimmed.len() > 1 {
        let lower = trimmed.to_lowercase();
        if !is_stop_word(&lower) {
            tokens.push(lower);
        }
    }
}

/// Split a single identifier into sub-tokens and push non-stop, non-short tokens.
///

// ---------------------------------------------------------------------------
// BM25 Inverted Index
// ---------------------------------------------------------------------------

/// BM25 index data for a single document.
#[derive(Clone)]
struct DocIndex {
    /// Normalised term frequencies: token -> (count / max_tf)
    tf: HashMap<String, f64>,
}

/// BM25-weighted inverted index.
pub struct Bm25Index {
    /// Per-document index: relative_path -> term frequency map
    docs: HashMap<String, DocIndex>,
    /// Document frequency: token -> number of documents containing it
    doc_freq: HashMap<String, usize>,
    /// Total number of indexed documents
    pub total_docs: usize,
    /// Average document length (in unique tokens)
    pub avgdl: f64,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

impl Bm25Index {
    pub fn new() -> Self {
        Bm25Index {
            docs: HashMap::new(),
            doc_freq: HashMap::new(),
            total_docs: 0,
            avgdl: 0.0,
        }
    }

    /// Add or update a document in the index.
    pub fn add_document(&mut self, path: &str, content: &str) {
        let tokens = tokenize(content);
        let mut tf: HashMap<String, f64> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_insert(0.0) += 1.0;
        }
        let max_tf = tf.values().cloned().fold(0.0_f64, f64::max).max(1.0);
        for v in tf.values_mut() {
            *v /= max_tf;
        }
        // If this path already exists, remove old doc freq contributions
        if let Some(old) = self.docs.remove(path) {
            for t in old.tf.keys() {
                if let Some(cnt) = self.doc_freq.get_mut(t) {
                    *cnt = cnt.saturating_sub(1);
                    if *cnt == 0 {
                        self.doc_freq.remove(t);
                    }
                }
            }
        }
        for t in tf.keys() {
            *self.doc_freq.entry(t.clone()).or_insert(0) += 1;
        }
        self.docs.insert(path.to_string(), DocIndex { tf });
        self.total_docs = self.docs.len();
        self._recompute_avgdl();
    }

    /// Remove a document from the index.
    pub fn remove_document(&mut self, path: &str) {
        if let Some(old) = self.docs.remove(path) {
            for t in old.tf.keys() {
                if let Some(cnt) = self.doc_freq.get_mut(t) {
                    *cnt = cnt.saturating_sub(1);
                    if *cnt == 0 {
                        self.doc_freq.remove(t);
                    }
                }
            }
        }
        self.total_docs = self.docs.len();
        self._recompute_avgdl();
    }

    /// Build the index from a batch of (path, content) pairs. Replaces all existing data.
    pub fn build(&mut self, files: Vec<(String, String)>) {
        self.docs.clear();
        self.doc_freq.clear();
        for (path, content) in &files {
            let tokens = tokenize(content);
            let mut tf: HashMap<String, f64> = HashMap::new();
            for t in &tokens {
                *tf.entry(t.clone()).or_insert(0.0) += 1.0;
            }
            let max_tf = tf.values().cloned().fold(0.0_f64, f64::max).max(1.0);
            for v in tf.values_mut() {
                *v /= max_tf;
            }
            for t in tf.keys() {
                *self.doc_freq.entry(t.clone()).or_insert(0) += 1;
            }
            self.docs.insert(path.clone(), DocIndex { tf });
        }
        self.total_docs = self.docs.len();
        self._recompute_avgdl();
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        self.docs.clear();
        self.doc_freq.clear();
        self.total_docs = 0;
        self.avgdl = 0.0;
    }

    /// Return the number of documents in the index.
    pub fn len(&self) -> usize {
        self.total_docs
    }

    pub fn is_empty(&self) -> bool {
        self.total_docs == 0
    }

    /// Search with Okapi BM25 ranking.
    ///
    /// Parameters:
    /// - `k1` = 1.5 (term frequency saturation)
    /// - `b` = 0.75 (length normalisation)
    /// - `name_bonus` = 2.0 (extra score when query token matches module path)
    pub fn search(
        &self,
        query: &str,
        limit: usize,
        k1: f64,
        b: f64,
        name_bonus: f64,
    ) -> Vec<(String, f64)> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() || self.total_docs == 0 {
            return vec![];
        }
        let avgdl = if self.avgdl > 0.0 { self.avgdl } else { 1.0 };
        let mut scores: Vec<(String, f64)> = Vec::with_capacity(self.docs.len());

        for (path, doc) in &self.docs {
            let mut score = 0.0;
            let dl = doc.tf.len() as f64;
            for qt in &query_tokens {
                let tf_val = match doc.tf.get(qt) {
                    Some(v) => *v,
                    None => continue,
                };
                let df = *self.doc_freq.get(qt).unwrap_or(&1) as f64;
                let idf = ((self.total_docs as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();
                let numerator = tf_val * (k1 + 1.0);
                let denominator = tf_val + k1 * (1.0 - b + b * dl / avgdl);
                score += idf * numerator / denominator.max(0.001);
            }
            // Name bonus: add when query token appears in module path
            let path_lower = path.to_lowercase();
            for qt in &query_tokens {
                if path_lower.contains(qt) {
                    score += name_bonus;
                }
            }
            if score > 0.0 {
                scores.push((path.clone(), score));
            }
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);
        scores
    }

    fn _recompute_avgdl(&mut self) {
        if self.docs.is_empty() {
            self.avgdl = 0.0;
            return;
        }
        let total: usize = self.docs.values().map(|d| d.tf.len()).sum();
        self.avgdl = total as f64 / self.docs.len() as f64;
    }
}
