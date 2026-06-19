//! Unified permission system for the EnTA agent framework.
//!
//! All authorization decisions — tool-level allow/deny/ask, dangerous-command
//! detection, and MCP tool resolution — are implemented here in Rust.  Python
//! tools call `permission_check` and forward the decision to the client; the
//! client may then call `permission_record_decision` to persist the answer
//! for future invocations.
//!
//! The three terminal states are:
//! * `Allow` — run the tool silently.
//! * `Deny`  — refuse, return a short error message to the model.
//! * `Ask`   — request user confirmation through the frontend.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{OnceLock, RwLock};

// ---------------------------------------------------------------------------
// Decision types
// ---------------------------------------------------------------------------

/// Outcome of a permission check.  Mirrors the protocol returned to Python.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "behavior", rename_all = "snake_case")]
pub enum PermissionDecision {
    /// Tool may execute without user interaction.
    Allow,
    /// Tool must not execute.
    Deny { reason: String },
    /// Tool may execute only after the user confirms.
    Ask { reason: String, rule: String },
}

impl PermissionDecision {
    pub fn behavior_str(&self) -> &'static str {
        match self {
            PermissionDecision::Allow => "allow",
            PermissionDecision::Deny { .. } => "deny",
            PermissionDecision::Ask { .. } => "ask",
        }
    }
}

// ---------------------------------------------------------------------------
// Policy table
// ---------------------------------------------------------------------------

/// Persisted user-managed policy for a tool or capability.  `Default` defers
/// to dangerous-pattern detection (only meaningful for the `bash` tool).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolicyValue {
    Allow,
    Deny,
    Ask,
    Default,
}

impl PolicyValue {
    fn from_str(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "allow" => PolicyValue::Allow,
            "deny" => PolicyValue::Deny,
            "ask" => PolicyValue::Ask,
            _ => PolicyValue::Default,
        }
    }
}

/// State holder for the active policy set.  Persists through Python.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PolicyState {
    /// Per-tool policy: `tool_name -> PolicyValue`.
    pub tools: HashMap<String, PolicyValue>,
    /// Per-capability policy: `network | docker | browser | agent | workflow | ...`.
    pub capabilities: HashMap<String, PolicyValue>,
}

// ---------------------------------------------------------------------------
// Tool capability table
// ---------------------------------------------------------------------------

/// Returns the set of capabilities a tool touches.  Empty for tools that need
/// no capability gate.  MCP tools keep the `mcp` capability so they can be
/// gated as a class.
pub fn tool_capabilities(name: &str) -> &'static [&'static str] {
    match name {
        "bash" | "bash_output" | "bash_kill" | "bash_list" => &["bash_io"],
        "file_read" | "file_write" | "file_edit" | "apply_patch" | "grep"
        | "glob" | "find_tool" => &["file"],
        "rest_client" => &["network"],
        "browser" => &["browser"],
        "docker" => &["docker"],
        "workflow" => &["workflow"],
        "git" => &["git"],
        "deploy" => &["deploy"],
        "desktop" | "computer_use" | "vlm_computer_use" => &["desktop"],
        "database" => &["database"],
        "notebook" | "pdf" | "spreadsheet" | "image" | "lsp" | "lint_format"
        | "test_runner" => &["misc"],
        "web_fetch" | "web_search" | "agent" | "cron_create" | "cron_delete"
        | "cron_list" | "task_create" | "task_get" | "task_list" | "task_update"
        | "task_stop" | "task_output" | "memory_create" | "memory_read"
        | "memory_update" | "memory_delete" | "memory_search" | "memory_profile"
        | "todo" | "question" => &[],
        _ => {
            if name.starts_with("mcp__") {
                &["mcp"]
            } else {
                &[]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dangerous-command patterns loaded from dangerous_commands.txt
// ---------------------------------------------------------------------------

/// Compiled regex table keyed by platform tag (`linux`, `windows`, `macos`).
#[derive(Default)]
struct DangerousTable {
    patterns: Vec<(String, Regex)>,
}

impl DangerousTable {
    fn load_from_file(path: &Path) -> Self {
        let mut patterns: Vec<(String, Regex)> = Vec::new();
        let Ok(text) = fs::read_to_string(path) else {
            return DangerousTable { patterns };
        };
        let mut current: Option<String> = None;
        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            // Bare section header lines such as ``[linux]``, ``[windows]``,
            // ``[macos]`` (with or without a leading '#') switch the
            // current platform tag.
            if line.starts_with('[') && line.ends_with(']') {
                let tag = line[1..line.len() - 1].trim().to_lowercase();
                if !tag.is_empty() {
                    current = Some(tag);
                }
                continue;
            }
            // Pure comment lines that look like ``# ── LINUX ──...`` are
            // ignored, but inline tags like ``# [linux]`` also switch the
            // section so older formatted files keep working.
            if let Some(rest) = line.strip_prefix('#') {
                if let Some(inner) = rest.trim().strip_prefix('[') {
                    if let Some(tag) = inner.strip_suffix(']') {
                        current = Some(tag.trim().to_lowercase());
                    }
                }
                continue;
            }
            let Some(tag) = current.clone() else {
                continue;
            };
            match Regex::new(line) {
                Ok(re) => patterns.push((tag.clone(), re)),
                Err(err) => eprintln!(
                    "[enta::permission] bad regex in dangerous_commands.txt \
                     under [{tag}]: {err}"
                ),
            }
        }
        DangerousTable { patterns }
    }

    /// Test a command against the patterns for the given platform.  Returns
    /// the first matching pattern string for use as a `rule` identifier.
    fn match_command(&self, command: &str, platform: &str) -> Option<String> {
        for (tag, re) in &self.patterns {
            if tag != platform {
                continue;
            }
            if re.is_match(command) {
                return Some(re.as_str().to_string());
            }
        }
        None
    }
}

fn current_platform_tag() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    }
}

// ---------------------------------------------------------------------------
// Manager — thread-safe global state
// ---------------------------------------------------------------------------

struct ManagerInner {
    state: PolicyState,
    dangerous: DangerousTable,
}

static MANAGER: OnceLock<RwLock<ManagerInner>> = OnceLock::new();

fn manager() -> &'static RwLock<ManagerInner> {
    MANAGER.get_or_init(|| {
        // Best-effort load of dangerous_commands.txt; not fatal if missing.
        let dangerous = find_dangerous_file()
            .map(|p| DangerousTable::load_from_file(&p))
            .unwrap_or_default();
        RwLock::new(ManagerInner {
            state: PolicyState::default(),
            dangerous,
        })
    })
}

fn find_dangerous_file() -> Option<std::path::PathBuf> {
    use std::env;

    // Probe a handful of well-known locations.  The most common case is
    // that the file ships next to the ``enta`` Python package, so we
    // look relative to the current working directory as well as a few
    // ancestor hops for when the agent runs from a project root.  We
    // also probe the executable's directory and the current working
    // directory's absolute path so packaged builds (PyInstaller, venv
    // activations) find the rules.
    let cwd = env::current_dir().ok();
    let exe = env::current_exe().ok();

    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    for raw in [
        "enta/dangerous_commands.txt",
        "dangerous_commands.txt",
        "../enta/dangerous_commands.txt",
        "../../enta/dangerous_commands.txt",
        "../../../enta/dangerous_commands.txt",
        "../../../../enta/dangerous_commands.txt",
    ] {
        candidates.push(std::path::PathBuf::from(raw));
    }
    if let Some(dir) = cwd.as_ref() {
        candidates.push(dir.join("enta/dangerous_commands.txt"));
        candidates.push(dir.join("dangerous_commands.txt"));
        candidates.push(dir.join("../enta/dangerous_commands.txt"));
        candidates.push(dir.join("../../enta/dangerous_commands.txt"));
        candidates.push(dir.join("../../../enta/dangerous_commands.txt"));
    }
    if let Some(exe_path) = exe.as_ref() {
        if let Some(parent) = exe_path.parent() {
            candidates.push(parent.join("enta/dangerous_commands.txt"));
            candidates.push(parent.join("../enta/dangerous_commands.txt"));
            candidates.push(parent.join("../../enta/dangerous_commands.txt"));
            candidates.push(parent.join("../../../enta/dangerous_commands.txt"));
            candidates.push(parent.join("../../../../enta/dangerous_commands.txt"));
        }
    }
    for c in &candidates {
        if c.exists() {
            return Some(c.clone());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Public API — called by the Python bindings
// ---------------------------------------------------------------------------

/// Replace the active policy table.  Used by the WS handler when the user
/// updates their permission settings.
pub fn permission_set_policies(state: PolicyState) {
    let mut g = manager().write().expect("permission lock poisoned");
    g.state = state;
}

/// Return a clone of the current policy table for sending to the client.
pub fn permission_get_policies() -> PolicyState {
    let g = manager().read().expect("permission lock poisoned");
    g.state.clone()
}

/// Record the user's answer to a previous `Ask` decision so we don't have to
/// ask again for the same tool.  `policy` is one of `allow` / `deny`.
pub fn permission_record_decision(tool_name: &str, policy: &str) {
    let mut g = manager().write().expect("permission lock poisoned");
    g.state
        .tools
        .insert(tool_name.to_string(), PolicyValue::from_str(policy));
}

// ---------------------------------------------------------------------------
// Always-allow list — tools that bypass ALL permission checks.
// These are non-dangerous tools (web fetch, search, memory, tasks, etc.)
// that should never prompt the user or be gated by a capability policy.
// ---------------------------------------------------------------------------

const ALWAYS_ALLOW: &[&str] = &[
    "web_fetch", "web_search", "agent",
    "cron_create", "cron_delete", "cron_list",
    "task_create", "task_get", "task_list", "task_update", "task_stop", "task_output",
    "memory_create", "memory_read", "memory_update", "memory_delete",
    "memory_search", "memory_profile",
    "todo", "question",
];

/// Core authorization check used by every tool before execution.
///
/// Precedence (highest first):
/// 0.  Always-allow list (these tools skip all checks entirely).
/// 1.  Tool-level override from the user policy table.
/// 2.  Capability-level override (`network`, `file`, ...).
/// 3.  Dangerous-pattern match for the `bash` tool family.
/// 4.  Default: allow.
pub fn permission_check(tool_name: &str, args: &str) -> PermissionDecision {
    // 0) Always-allow tools skip everything.
    if ALWAYS_ALLOW.contains(&tool_name) {
        return PermissionDecision::Allow;
    }

    let g = manager().read().expect("permission lock poisoned");

    // 1) Tool-level override.
    if let Some(p) = g.state.tools.get(tool_name) {
        return decision_from_policy(p, tool_name);
    }

    // 2) Capability-level override.
    for cap in tool_capabilities(tool_name) {
        if let Some(p) = g.state.capabilities.get(*cap) {
            return decision_from_policy(p, tool_name);
        }
    }

    // 3) Bash dangerous patterns.
    if matches!(tool_name, "bash" | "bash_output") {
        if let Some(rule) = g.dangerous.match_command(args, current_platform_tag()) {
            return PermissionDecision::Ask {
                reason: format!(
                    "Command matches dangerous pattern: {rule}"
                ),
                rule,
            };
        }
    }

    PermissionDecision::Allow
}

fn decision_from_policy(p: &PolicyValue, tool_name: &str) -> PermissionDecision {
    match p {
        PolicyValue::Allow => PermissionDecision::Allow,
        PolicyValue::Deny => PermissionDecision::Deny {
            reason: format!("Tool '{tool_name}' is denied by user policy."),
        },
        PolicyValue::Ask => PermissionDecision::Ask {
            reason: format!("Tool '{tool_name}' is configured to ask for permission."),
            rule: "user_ask".to_string(),
        },
        PolicyValue::Default => PermissionDecision::Allow,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capabilities() {
        assert!(tool_capabilities("bash").contains(&"bash_io"));
        assert!(tool_capabilities("file_read").contains(&"file"));
        assert!(tool_capabilities("mcp__fs__read").contains(&"mcp"));
        assert!(tool_capabilities("todo").contains(&"misc"));
    }

    #[test]
    fn test_default_allows() {
        assert_eq!(permission_check("todo", "{}").behavior_str(), "allow");
    }

    #[test]
    fn test_deny_short_circuits() {
        let mut state = PolicyState::default();
        state
            .tools
            .insert("todo".to_string(), PolicyValue::Deny);
        permission_set_policies(state);
        match permission_check("todo", "{}") {
            PermissionDecision::Deny { .. } => {}
            other => panic!("expected deny, got {other:?}"),
        }
        // Restore default for other tests.
        permission_set_policies(PolicyState::default());
    }
}
