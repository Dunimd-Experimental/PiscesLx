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

use similar::{ChangeTag, TextDiff};

pub fn compute_diff(old: &str, new: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut result = String::new();

    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            ChangeTag::Delete => "-",
            ChangeTag::Insert => "+",
            ChangeTag::Equal => " ",
        };
        result.push_str(&format!("{}{}", sign, change.value()));
    }

    result
}

pub fn apply_diff(content: &str, diff: &str) -> Result<String, String> {
    let mut result = String::new();
    let content_lines: Vec<&str> = content.lines().collect();
    let mut content_idx = 0;

    for line in diff.lines() {
        if line.is_empty() {
            continue;
        }
        let (tag, rest) = line.split_at(1);
        match tag {
            " " | "" => {
                if content_idx < content_lines.len() {
                    result.push_str(content_lines[content_idx]);
                    result.push('\n');
                    content_idx += 1;
                } else {
                    result.push_str(rest);
                    result.push('\n');
                }
            }
            "+" => {
                result.push_str(rest);
                result.push('\n');
            }
            "-" => {
                content_idx += 1;
            }
            _ => {
                if content_idx < content_lines.len() {
                    result.push_str(content_lines[content_idx]);
                    result.push('\n');
                    content_idx += 1;
                }
            }
        }
    }

    while content_idx < content_lines.len() {
        result.push_str(content_lines[content_idx]);
        result.push('\n');
        content_idx += 1;
    }

    Ok(result)
}
