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

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// LSP domain types
// ---------------------------------------------------------------------------

/// A zero-based position in a text document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspPosition {
    /// Zero-based line index.
    pub line: u32,
    /// Zero-based UTF-16 code-unit offset on the line.
    pub character: u32,
}

/// A range between two positions in a text document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspRange {
    pub start: LspPosition,
    pub end: LspPosition,
}

/// A diagnostic reported by a language server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspDiagnostic {
    /// The range at which the diagnostic applies.
    pub range: LspRange,
    /// Severity: 1 = Error, 2 = Warning, 3 = Information, 4 = Hint.
    #[serde(default)]
    pub severity: u8,
    /// Human-readable diagnostic message.
    #[serde(default)]
    pub message: String,
    /// Source of the diagnostic (e.g. "rustc", "pyright").
    #[serde(default)]
    pub source: String,
}

/// A generic JSON-RPC 2.0 message (request or notification).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspMessage {
    /// Must be "2.0".
    #[serde(default = "default_jsonrpc")]
    pub jsonrpc: String,
    /// Request ID (`None` for notifications).
    #[serde(default)]
    pub id: Option<u64>,
    /// Method name (e.g. "textDocument/publishDiagnostics").
    #[serde(default)]
    pub method: String,
    /// Method parameters as raw JSON.
    #[serde(default)]
    pub params: serde_json::Value,
}

fn default_jsonrpc() -> String {
    "2.0".to_string()
}

/// Wrapper used when parsing `textDocument/publishDiagnostics` notifications.
#[derive(Deserialize)]
struct PublishDiagnosticsParams {
    #[allow(dead_code)]
    uri: String,
    diagnostics: Vec<LspDiagnostic>,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse a raw JSON-RPC message string into an `LspMessage`.
///
/// The message must be a valid JSON-RPC 2.0 request or notification.
/// The Content-Length header (if present) should already be stripped
/// by the caller.
pub fn parse_lsp_message(raw: &str) -> Result<LspMessage, String> {
    serde_json::from_str::<LspMessage>(raw)
        .map_err(|e| format!("Failed to parse LSP message: {}", e))
}

/// Extract diagnostics from a `textDocument/publishDiagnostics` notification
/// payload.
///
/// `raw` should be the **params** object of the notification (i.e. the
/// `params` field from the JSON-RPC envelope).
pub fn parse_diagnostics(raw: &str) -> Result<Vec<LspDiagnostic>, String> {
    let params: PublishDiagnosticsParams = serde_json::from_str(raw)
        .map_err(|e| format!("Failed to parse publishDiagnostics params: {}", e))?;
    Ok(params.diagnostics)
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Build a JSON-RPC 2.0 request string.
///
/// Produces a compact JSON string suitable for sending over stdout/stdin
/// to an LSP server.  The caller should prepend a Content-Length header
/// (see [`build_content_length_header`]).
pub fn build_lsp_request(id: u64, method: &str, params: serde_json::Value) -> String {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    });
    // Compact representation: no extra whitespace
    serde_json::to_string(&msg).unwrap_or_else(|_| r#"{"jsonrpc":"2.0"}"#.to_string())
}

/// Build a JSON-RPC 2.0 notification string (no `id` field).
pub fn build_lsp_notification(method: &str, params: serde_json::Value) -> String {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    });
    serde_json::to_string(&msg).unwrap_or_else(|_| r#"{"jsonrpc":"2.0"}"#.to_string())
}

/// Build an LSP HTTP-style `Content-Length` header for the given body.
///
/// The LSP protocol transmits messages over stdout/stdin with a
/// `Content-Length: <N>\r\n\r\n` prefix followed by the JSON body.
pub fn build_content_length_header(content: &str) -> String {
    format!("Content-Length: {}\r\n\r\n", content.len())
}

/// Wrap a full LSP message (header + body) ready for transmission.
///
/// This is a convenience that calls [`build_content_length_header`] and
/// appends the JSON body.
pub fn build_lsp_message(body: &str) -> String {
    format!("{}{}", build_content_length_header(body), body)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_lsp_message() {
        let raw = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let msg = parse_lsp_message(raw).unwrap();
        assert_eq!(msg.jsonrpc, "2.0");
        assert_eq!(msg.id, Some(1));
        assert_eq!(msg.method, "initialize");
    }

    #[test]
    fn test_parse_notification() {
        let raw = r#"{"jsonrpc":"2.0","method":"exit","params":null}"#;
        let msg = parse_lsp_message(raw).unwrap();
        assert!(msg.id.is_none());
        assert_eq!(msg.method, "exit");
    }

    #[test]
    fn test_parse_diagnostics() {
        let raw = r#"{
            "uri": "file:///test.py",
            "diagnostics": [
                {
                    "range": {
                        "start": {"line": 1, "character": 0},
                        "end": {"line": 1, "character": 10}
                    },
                    "severity": 1,
                    "message": "Syntax error",
                    "source": "pyright"
                }
            ]
        }"#;
        let diags = parse_diagnostics(raw).unwrap();
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].severity, 1);
        assert_eq!(diags[0].message, "Syntax error");
        assert_eq!(diags[0].source, "pyright");
        assert_eq!(diags[0].range.start.line, 1);
    }

    #[test]
    fn test_build_lsp_request() {
        let req = build_lsp_request(42, "textDocument/completion", serde_json::json!({}));
        assert!(req.contains("42"));
        assert!(req.contains("textDocument/completion"));
        assert!(req.starts_with('{'));
    }

    #[test]
    fn test_content_length_header() {
        let header = build_content_length_header("hello");
        assert_eq!(header, "Content-Length: 5\r\n\r\n");
    }

    #[test]
    fn test_build_lsp_message() {
        let body = r#"{"jsonrpc":"2.0","method":"test"}"#;
        let msg = build_lsp_message(body);
        assert!(msg.starts_with("Content-Length:"));
        assert!(msg.ends_with(body));
    }
}
