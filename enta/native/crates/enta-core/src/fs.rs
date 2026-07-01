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

use std::path::Path;

pub fn read_file(path: &str, offset: usize, limit: usize) -> Result<String, String> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(format!("File not found: {}", path));
    }

    let content = std::fs::read_to_string(p).map_err(|e| format!("Error reading file: {}", e))?;

    let lines: Vec<&str> = content.lines().collect();
    let start = if offset > 0 { offset - 1 } else { 0 };
    let end = if limit > 0 {
        std::cmp::min(start + limit, lines.len())
    } else {
        lines.len()
    };

    if start >= lines.len() {
        return Ok(String::new());
    }

    Ok(lines[start..end].join("\n"))
}

pub fn write_file(path: &str, content: &str) -> Result<bool, String> {
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Error creating directories: {}", e))?;
    }
    std::fs::write(p, content).map_err(|e| format!("Error writing file: {}", e))?;
    Ok(true)
}
