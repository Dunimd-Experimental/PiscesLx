//! Copyright (c) 2025-2026 Wenze Wei. All Rights Reserved.
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

use pyo3::prelude::*;
use pyo3::types::PyDict;
use enta::diff;
use enta::embedding;
use enta::fs;
use enta::indexer;
use enta::landlock;
use enta::lsp_proto;
use enta::permission;
use enta::sandbox;
use enta::search;
use enta::shell;
use enta::simd_search;
use enta::tokenizer;

// ---------------------------------------------------------------------------
// Helpers — convert Rust structs to Python dicts for zero-copy perf
// ---------------------------------------------------------------------------

fn search_results_to_py(py: Python, results: &[search::SearchResult]) -> PyResult<Vec<PyObject>> {
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        let d = PyDict::new(py);
        d.set_item("file_path", &r.file_path)?;
        d.set_item("line_number", r.line_number)?;
        d.set_item("line_content", &r.line_content)?;
        d.set_item("score", r.score)?;
        out.push(d.into());
    }
    Ok(out)
}

fn sandbox_result_to_py(py: Python, r: &sandbox::SandboxResult) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("stdout", &r.stdout)?;
    d.set_item("stderr", &r.stderr)?;
    d.set_item("exit_code", r.exit_code)?;
    Ok(d.into())
}

fn shell_result_to_py(py: Python, r: &shell::ShellResult) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("stdout", &r.stdout)?;
    d.set_item("stderr", &r.stderr)?;
    d.set_item("exit_code", r.exit_code)?;
    Ok(d.into())
}

fn lsp_diagnostics_to_py(py: Python, diags: &[lsp_proto::LspDiagnostic]) -> PyResult<Vec<PyObject>> {
    let mut out = Vec::with_capacity(diags.len());
    for diag in diags {
        let d = PyDict::new(py);

        let range = PyDict::new(py);
        {
            let start = PyDict::new(py);
            start.set_item("line", diag.range.start.line)?;
            start.set_item("character", diag.range.start.character)?;
            range.set_item("start", start)?;

            let end = PyDict::new(py);
            end.set_item("line", diag.range.end.line)?;
            end.set_item("character", diag.range.end.character)?;
            range.set_item("end", end)?;
        }
        d.set_item("range", range)?;
        d.set_item("severity", diag.severity)?;
        d.set_item("message", &diag.message)?;
        d.set_item("source", &diag.source)?;
        out.push(d.into());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Existing Python-facing functions
// ---------------------------------------------------------------------------

#[pyfunction]
fn search_codebase(py: Python, query: &str, path: Option<&str>) -> PyResult<Vec<PyObject>> {
    let p = path.unwrap_or(".");
    let results = search::search_codebase(query, p);
    search_results_to_py(py, &results)
}

#[pyfunction]
#[pyo3(signature = (path, offset=0, limit=0))]
fn read_file(path: &str, offset: usize, limit: usize) -> PyResult<String> {
    fs::read_file(path, offset, limit)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

#[pyfunction]
fn write_file(path: &str, content: &str) -> PyResult<bool> {
    fs::write_file(path, content).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

#[pyfunction]
#[pyo3(signature = (pattern, path, case_insensitive=false, glob_filter=None))]
fn grep(
    py: Python,
    pattern: &str,
    path: &str,
    case_insensitive: bool,
    glob_filter: Option<&str>,
) -> PyResult<Vec<PyObject>> {
    let results = search::grep(pattern, path, case_insensitive, glob_filter);
    search_results_to_py(py, &results)
}

#[pyfunction]
fn glob(pattern: &str, path: Option<&str>) -> PyResult<Vec<String>> {
    let p = path.unwrap_or(".");
    Ok(search::glob(pattern, p))
}

#[pyfunction]
fn count_tokens(text: &str) -> usize {
    tokenizer::count_tokens(text)
}

#[pyfunction]
fn compute_diff(old: &str, new: &str) -> String {
    diff::compute_diff(old, new)
}

#[pyfunction]
fn apply_diff(content: &str, diff_str: &str) -> PyResult<String> {
    diff::apply_diff(content, diff_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

#[pyfunction]
#[pyo3(signature = (command, timeout=30, workspace=None))]
fn sandbox_execute(
    py: Python,
    command: &str,
    timeout: u64,
    workspace: Option<&str>,
) -> PyResult<PyObject> {
    let result = sandbox::sandbox_execute(command, timeout, workspace)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    sandbox_result_to_py(py, &result)
}

#[pyfunction]
fn sandbox_read_file(path: &str) -> PyResult<String> {
    sandbox::sandbox_read_file(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

#[pyfunction]
fn sandbox_write_file(path: &str, content: &str) -> PyResult<bool> {
    sandbox::sandbox_write_file(path, content)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

#[pyfunction]
fn execute_shell(py: Python, command: &str, cwd: Option<&str>, timeout: Option<u64>) -> PyResult<PyObject> {
    let t = timeout.unwrap_or(30);
    let result = shell::execute_shell(command, cwd, t)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    shell_result_to_py(py, &result)
}

// ---------------------------------------------------------------------------
// New: embedding.rs bindings
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two f32 slices.
#[pyfunction]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> f32 {
    embedding::cosine_similarity(&a, &b)
}

/// Compute Jaccard text similarity on whitespace-delimited tokens.
#[pyfunction]
fn text_similarity(a: &str, b: &str) -> f32 {
    embedding::text_similarity(a, b)
}

// ---------------------------------------------------------------------------
// New: simd_search.rs bindings
// ---------------------------------------------------------------------------

/// SIMD-accelerated substring check.
#[pyfunction]
fn simd_contains(haystack: &str, needle: &str) -> bool {
    simd_search::simd_contains(haystack, needle)
}

/// SIMD-accelerated find-all match positions (byte offsets).
#[pyfunction]
fn simd_find_all(haystack: &str, needle: &str) -> Vec<usize> {
    simd_search::simd_find_all(haystack, needle)
}

/// SIMD-accelerated byte-level memmem.
#[pyfunction]
fn simd_memmem(haystack: Vec<u8>, needle: Vec<u8>) -> Option<usize> {
    simd_search::simd_memmem(&haystack, &needle)
}

// ---------------------------------------------------------------------------
// New: landlock.rs bindings
// ---------------------------------------------------------------------------

/// Restrict the current thread to read-only filesystem access under paths.
#[pyfunction]
fn landlock_restrict_read_only(paths: Vec<String>) -> PyResult<()> {
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    landlock::landlock_restrict_read_only(&refs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Restrict the current thread from making network connections.
#[pyfunction]
fn landlock_restrict_network() -> PyResult<()> {
    landlock::landlock_restrict_network()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Full sandbox: read-only filesystem under workspace, no network, no exec.
#[pyfunction]
fn landlock_full_sandbox(workspace: &str) -> PyResult<()> {
    landlock::landlock_full_sandbox(workspace)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Workspace sandbox: read-write access under workspace, no network, no exec.
#[pyfunction]
fn landlock_workspace_sandbox(workspace: &str) -> PyResult<()> {
    landlock::landlock_workspace_sandbox(workspace)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Check whether Landlock is available on the current kernel.
#[pyfunction]
fn landlock_available() -> bool {
    landlock::landlock_available()
}

/// Return the highest Landlock ABI version (0 if not available).
#[pyfunction]
fn landlock_abi_version() -> u64 {
    landlock::landlock_abi_version()
}

// ---------------------------------------------------------------------------
// New: lsp_proto.rs bindings
// ---------------------------------------------------------------------------

/// Parse a raw JSON-RPC 2.0 message string into a dict.
#[pyfunction]
fn parse_lsp_message(raw: &str) -> PyResult<String> {
    // Return the parsed message as a JSON string; Python can json.loads it.
    let msg = lsp_proto::parse_lsp_message(raw)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    serde_json::to_string(&msg)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Extract diagnostics from a publishDiagnostics notification params.
#[pyfunction]
fn parse_diagnostics(py: Python, raw: &str) -> PyResult<Vec<PyObject>> {
    let diags = lsp_proto::parse_diagnostics(raw)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    lsp_diagnostics_to_py(py, &diags)
}

/// Build a JSON-RPC 2.0 request string.
#[pyfunction]
fn build_lsp_request(id: u64, method: &str, params: &str) -> PyResult<String> {
    let params_val: serde_json::Value = serde_json::from_str(params)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(lsp_proto::build_lsp_request(id, method, params_val))
}

/// Build an LSP Content-Length header for the given body.
#[pyfunction]
fn build_content_length_header(content: &str) -> String {
    lsp_proto::build_content_length_header(content)
}

// ---------------------------------------------------------------------------
// BM25 Indexer (PyO3 class)
// ---------------------------------------------------------------------------

/// BM25-powered code search index.
///
/// Usage from Python::
///
///     from enta._native import Bm25Index
///     idx = Bm25Index()
///     idx.build([("path.py", "def foo(): pass")])
///     results = idx.search("foo", 10)
#[pyclass]
struct Bm25Index {
    inner: indexer::Bm25Index,
}

#[pymethods]
impl Bm25Index {
    #[new]
    fn new() -> Self {
        Bm25Index {
            inner: indexer::Bm25Index::new(),
        }
    }

    /// Build or rebuild the index from a list of (path, content) pairs.
    fn build(&mut self, files: Vec<(String, String)>) {
        self.inner.build(files);
    }

    /// Search with BM25 ranking. Returns list of (path, score).
    fn search(&self, query: &str, limit: Option<usize>) -> Vec<(String, f64)> {
        self.inner
            .search(query, limit.unwrap_or(10), 1.5, 0.75, 2.0)
    }

    /// Add or update a single document.
    fn add_document(&mut self, path: &str, content: &str) {
        self.inner.add_document(path, content);
    }

    /// Remove a document by path.
    fn remove_document(&mut self, path: &str) {
        self.inner.remove_document(path);
    }

    /// Clear the entire index.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Number of indexed documents.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __bool__(&self) -> bool {
        !self.inner.is_empty()
    }
}

// ---------------------------------------------------------------------------
// New: permission.rs bindings
// ---------------------------------------------------------------------------

/// Run the unified permission check.  Returns a dict of the form
/// ``{"behavior": "allow"|"deny"|"ask", "reason": str?, "rule": str?}``.
#[pyfunction]
fn permission_check(py: Python, tool_name: &str, args: &str) -> PyResult<PyObject> {
    let d = permission::permission_check(tool_name, args);
    let out = PyDict::new(py);
    out.set_item("behavior", d.behavior_str())?;
    match d {
        permission::PermissionDecision::Allow => {}
        permission::PermissionDecision::Deny { reason } => {
            out.set_item("reason", reason)?;
        }
        permission::PermissionDecision::Ask { reason, rule } => {
            out.set_item("reason", reason)?;
            out.set_item("rule", rule)?;
        }
    }
    Ok(out.into())
}

/// Replace the active permission policy table.  ``policies_json`` is a JSON
/// string ``{"tools": {...}, "capabilities": {...}}`` where each value is
/// one of ``"allow" | "deny" | "ask" | "default"``.
#[pyfunction]
fn permission_set_policies(policies_json: &str) -> PyResult<()> {
    let state: permission::PolicyState = serde_json::from_str(policies_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    permission::permission_set_policies(state);
    Ok(())
}

/// Return the active permission policy table as a JSON string.
#[pyfunction]
fn permission_get_policies() -> PyResult<String> {
    let state = permission::permission_get_policies();
    serde_json::to_string(&state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Persist the user's answer to a previous ``Ask`` decision so it doesn't
/// have to be asked again for the same tool.
#[pyfunction]
fn permission_record_decision(tool_name: &str, policy: &str) -> PyResult<()> {
    permission::permission_record_decision(tool_name, policy);
    Ok(())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _native(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Existing functions
    m.add_function(wrap_pyfunction!(search_codebase, m)?)?;
    m.add_function(wrap_pyfunction!(read_file, m)?)?;
    m.add_function(wrap_pyfunction!(write_file, m)?)?;
    m.add_function(wrap_pyfunction!(grep, m)?)?;
    m.add_function(wrap_pyfunction!(glob, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(compute_diff, m)?)?;
    m.add_function(wrap_pyfunction!(apply_diff, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox_execute, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox_read_file, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox_write_file, m)?)?;
    m.add_function(wrap_pyfunction!(execute_shell, m)?)?;

    // New: embedding
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(text_similarity, m)?)?;

    // New: simd_search
    m.add_function(wrap_pyfunction!(simd_contains, m)?)?;
    m.add_function(wrap_pyfunction!(simd_find_all, m)?)?;
    m.add_function(wrap_pyfunction!(simd_memmem, m)?)?;

    // New: landlock
    m.add_function(wrap_pyfunction!(landlock_restrict_read_only, m)?)?;
    m.add_function(wrap_pyfunction!(landlock_restrict_network, m)?)?;
    m.add_function(wrap_pyfunction!(landlock_full_sandbox, m)?)?;
    m.add_function(wrap_pyfunction!(landlock_workspace_sandbox, m)?)?;
    m.add_function(wrap_pyfunction!(landlock_available, m)?)?;
    m.add_function(wrap_pyfunction!(landlock_abi_version, m)?)?;

    // New: lsp_proto
    m.add_function(wrap_pyfunction!(parse_lsp_message, m)?)?;
    m.add_function(wrap_pyfunction!(parse_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(build_lsp_request, m)?)?;
    m.add_function(wrap_pyfunction!(build_content_length_header, m)?)?;

    // New: permission system
    m.add_function(wrap_pyfunction!(permission_check, m)?)?;
    m.add_function(wrap_pyfunction!(permission_set_policies, m)?)?;
    m.add_function(wrap_pyfunction!(permission_get_policies, m)?)?;
    m.add_function(wrap_pyfunction!(permission_record_decision, m)?)?;

    // BM25 indexer
    m.add_class::<Bm25Index>()?;

    Ok(())
}
