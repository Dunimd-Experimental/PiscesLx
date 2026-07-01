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

use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

#[derive(Serialize, Deserialize, Debug)]
pub struct ShellResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn execute_shell(
    command: &str,
    cwd: Option<&str>,
    timeout_secs: u64,
) -> Result<ShellResult, String> {
    #[cfg(target_os = "windows")]
    {
        let mut cmd = std::process::Command::new("cmd.exe");
        cmd.arg("/U");  // UTF-16LE output — preserves all Unicode
        cmd.arg("/C");
        cmd.arg(command);
        use std::os::windows::process::CommandExt as _;
        cmd.creation_flags(0x08000000);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(cmd.output());
        });
        let output = rx
            .recv_timeout(Duration::from_secs(timeout_secs))
            .map_err(|_| format!("Timed out after {timeout_secs}s"))?
            .map_err(|e| format!("{e}"))?;
        return Ok(ShellResult {
            stdout: decode_win(&output.stdout),
            stderr: decode_win(&output.stderr),
            exit_code: output.status.code().unwrap_or(-1),
        });
    }

    #[cfg(not(target_os = "windows"))]
    {
        let mut cmd = std::process::Command::new("sh");
        cmd.arg("-c");
        cmd.arg(command);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(cmd.output());
        });
        let output = rx
            .recv_timeout(Duration::from_secs(timeout_secs))
            .map_err(|_| format!("Timed out after {timeout_secs}s"))?
            .map_err(|e| format!("{e}"))?;
        Ok(ShellResult {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}

#[cfg(target_os = "windows")]
fn decode_win(raw: &[u8]) -> String {
    // ``cmd /U`` produces UTF-16LE for built-in commands (echo, dir, etc.)
    // but external programs (git, python, node) output their own encoding,
    // typically ASCII or UTF-8.  Detect UTF-16LE by checking whether every
    // other byte is zero — the hallmark of ASCII in UTF-16LE encoding.
    let looks_utf16 = raw.len() >= 2 && raw.len() % 2 == 0 && {
        // Check the first ~40 u16 values: if every high byte is zero,
        // this is ASCII output from cmd /U (which produces UTF-16LE).
        // External tools (git, python) ignore /U and produce UTF-8/ASCII,
        // where the first 40 bytes won't have this alternating-zero pattern.
        let min_len = raw.len().min(80);
        (0..min_len).step_by(2).all(|i| raw[i + 1] == 0)
    };

    if looks_utf16 {
        let u16: Vec<u16> = raw
            .windows(2)
            .step_by(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16_lossy(&u16)
    } else {
        String::from_utf8_lossy(raw).into_owned()
    }
}
