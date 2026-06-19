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

// ---------------------------------------------------------------------------
// SIMD helpers (feature-gated)
// ---------------------------------------------------------------------------

/// Size of a single SIMD lane in bytes.
#[allow(dead_code)]
const SIMD_LANE: usize = 32;

#[cfg(feature = "simd")]
mod simd_impl {
    use wide::u8x32;

    /// Return true when `haystack[i..]` starts with `needle`.
    #[inline(always)]
    fn matches_at(haystack: &[u8], needle: &[u8], pos: usize) -> bool {
        haystack[pos..].starts_with(needle)
    }

    /// SIMD-accelerated substring check.
    pub fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.len() > haystack.len() {
            return false;
        }
        if needle.len() < 8 {
            return haystack.windows(needle.len()).any(|w| w == needle);
        }

        let first = u8x32::splat(needle[0]);
        let last_pos = haystack.len() - needle.len();
        let mut i = 0usize;

        while i <= last_pos {
            let remaining = last_pos - i + 1;
            if remaining < super::SIMD_LANE {
                return haystack[i..].windows(needle.len()).any(|w| w == needle);
            }

            // Load 32 bytes and compare first byte in parallel
            let chunk = u8x32::new([
                haystack[i], haystack[i+1], haystack[i+2], haystack[i+3],
                haystack[i+4], haystack[i+5], haystack[i+6], haystack[i+7],
                haystack[i+8], haystack[i+9], haystack[i+10], haystack[i+11],
                haystack[i+12], haystack[i+13], haystack[i+14], haystack[i+15],
                haystack[i+16], haystack[i+17], haystack[i+18], haystack[i+19],
                haystack[i+20], haystack[i+21], haystack[i+22], haystack[i+23],
                haystack[i+24], haystack[i+25], haystack[i+26], haystack[i+27],
                haystack[i+28], haystack[i+29], haystack[i+30], haystack[i+31],
            ]);
            let mask = chunk.cmp_eq(first);
            let bits = mask.to_bitmask();

            if bits != 0 {
                for j in 0..super::SIMD_LANE {
                    if (bits >> j) & 1 != 0 {
                        let pos = i + j;
                        if pos <= last_pos && matches_at(haystack, needle, pos) {
                            return true;
                        }
                    }
                }
            }
            i += super::SIMD_LANE;
        }
        false
    }

    /// SIMD-accelerated find-all positions.
    pub fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
        let mut positions = Vec::new();
        if needle.is_empty() || needle.len() > haystack.len() {
            return positions;
        }
        if needle.len() < 8 {
            for (i, w) in haystack.windows(needle.len()).enumerate() {
                if w == needle {
                    positions.push(i);
                }
            }
            return positions;
        }

        let first = u8x32::splat(needle[0]);
        let last_pos = haystack.len() - needle.len();
        let mut i = 0usize;

        while i <= last_pos {
            let remaining = last_pos - i + 1;
            if remaining < super::SIMD_LANE {
                for (j, w) in haystack[i..].windows(needle.len()).enumerate() {
                    if w == needle {
                        positions.push(i + j);
                    }
                }
                break;
            }

            let chunk = u8x32::new([
                haystack[i], haystack[i+1], haystack[i+2], haystack[i+3],
                haystack[i+4], haystack[i+5], haystack[i+6], haystack[i+7],
                haystack[i+8], haystack[i+9], haystack[i+10], haystack[i+11],
                haystack[i+12], haystack[i+13], haystack[i+14], haystack[i+15],
                haystack[i+16], haystack[i+17], haystack[i+18], haystack[i+19],
                haystack[i+20], haystack[i+21], haystack[i+22], haystack[i+23],
                haystack[i+24], haystack[i+25], haystack[i+26], haystack[i+27],
                haystack[i+28], haystack[i+29], haystack[i+30], haystack[i+31],
            ]);
            let mask = chunk.cmp_eq(first);
            let bits = mask.to_bitmask();

            if bits != 0 {
                for j in 0..super::SIMD_LANE {
                    if (bits >> j) & 1 != 0 {
                        let pos = i + j;
                        if pos <= last_pos && matches_at(haystack, needle, pos) {
                            positions.push(pos);
                        }
                    }
                }
            }
            i += super::SIMD_LANE;
        }
        positions
    }

    /// SIMD-accelerated memmem (byte-level search).
    pub fn memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if needle.len() > haystack.len() {
            return None;
        }
        if needle.len() < 8 {
            return haystack.windows(needle.len()).position(|w| w == needle);
        }

        let first = u8x32::splat(needle[0]);
        let last_pos = haystack.len() - needle.len();
        let mut i = 0usize;

        while i <= last_pos {
            let remaining = last_pos - i + 1;
            if remaining < super::SIMD_LANE {
                return haystack[i..].windows(needle.len()).position(|w| w == needle).map(|p| i + p);
            }

            let chunk = u8x32::new([
                haystack[i], haystack[i+1], haystack[i+2], haystack[i+3],
                haystack[i+4], haystack[i+5], haystack[i+6], haystack[i+7],
                haystack[i+8], haystack[i+9], haystack[i+10], haystack[i+11],
                haystack[i+12], haystack[i+13], haystack[i+14], haystack[i+15],
                haystack[i+16], haystack[i+17], haystack[i+18], haystack[i+19],
                haystack[i+20], haystack[i+21], haystack[i+22], haystack[i+23],
                haystack[i+24], haystack[i+25], haystack[i+26], haystack[i+27],
                haystack[i+28], haystack[i+29], haystack[i+30], haystack[i+31],
            ]);
            let mask = chunk.cmp_eq(first);
            let bits = mask.to_bitmask();

            if bits != 0 {
                let tz = bits.trailing_zeros() as usize;
                if tz < super::SIMD_LANE {
                    let pos = i + tz;
                    if pos <= last_pos && matches_at(haystack, needle, pos) {
                        return Some(pos);
                    }
                }
                // Check remaining bits
                for j in tz+1..super::SIMD_LANE {
                    if (bits >> j) & 1 != 0 {
                        let pos = i + j;
                        if pos <= last_pos && matches_at(haystack, needle, pos) {
                            return Some(pos);
                        }
                    }
                }
            }
            i += super::SIMD_LANE;
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback (no SIMD feature)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "simd"))]
mod scalar_impl {
    /// Process 32 bytes at a time by scanning for the first byte match,
    /// then verifying the full needle.  This is algorithmically similar to
    /// the SIMD path but uses a tight scalar loop.
    #[inline]
    pub fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.len() > haystack.len() {
            return false;
        }
        if needle.len() < 8 {
            return haystack.windows(needle.len()).any(|w| w == needle);
        }
        super::find_first(haystack, needle).is_some()
    }

    #[inline]
    pub fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
        let mut positions = Vec::new();
        if needle.is_empty() || needle.len() > haystack.len() {
            return positions;
        }
        if needle.len() < 8 {
            for (i, w) in haystack.windows(needle.len()).enumerate() {
                if w == needle {
                    positions.push(i);
                }
            }
            return positions;
        }
        let first = needle[0];
        let last_pos = haystack.len() - needle.len();
        for i in 0..=last_pos {
            if haystack[i] == first && haystack[i..].starts_with(needle) {
                positions.push(i);
            }
        }
        positions
    }

    #[inline]
    pub fn memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if needle.len() > haystack.len() {
            return None;
        }
        if needle.len() < 8 {
            return haystack.windows(needle.len()).position(|w| w == needle);
        }
        super::find_first(haystack, needle)
    }
}

/// Shared scalar helper: find the first occurrence of `needle` in `haystack`
/// by scanning for the first byte and verifying.
#[inline]
fn find_first(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let first = needle[0];
    let last_pos = haystack.len() - needle.len();
    for i in 0..=last_pos {
        if haystack[i] == first && haystack[i..].starts_with(needle) {
            return Some(i);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Public API — delegates to the active implementation
// ---------------------------------------------------------------------------

/// Check whether `haystack` contains `needle` using SIMD-accelerated search.
///
/// Falls back to scalar search when the needle is shorter than 8 bytes.
/// Empty needle returns `true`.
pub fn simd_contains(haystack: &str, needle: &str) -> bool {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() {
        return true;
    }

    #[cfg(feature = "simd")]
    {
        simd_impl::contains(h, n)
    }
    #[cfg(not(feature = "simd"))]
    {
        scalar_impl::contains(h, n)
    }
}

/// Return all byte-offset positions where `needle` occurs in `haystack`.
///
/// Uses SIMD to scan the first byte in parallel, then verifies each
/// candidate.  Returns an empty `Vec` when the needle is empty.
pub fn simd_find_all(haystack: &str, needle: &str) -> Vec<usize> {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();

    #[cfg(feature = "simd")]
    {
        simd_impl::find_all(h, n)
    }
    #[cfg(not(feature = "simd"))]
    {
        scalar_impl::find_all(h, n)
    }
}

/// Byte-level substring search (`memmem`).
///
/// Returns `Some(offset)` of the first match or `None` when not found.
/// An empty needle returns `Some(0)`.
pub fn simd_memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    #[cfg(feature = "simd")]
    {
        simd_impl::memmem(haystack, needle)
    }
    #[cfg(not(feature = "simd"))]
    {
        scalar_impl::memmem(haystack, needle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_basic() {
        assert!(simd_contains("hello world", "world"));
        assert!(!simd_contains("hello world", "planet"));
        assert!(simd_contains("hello world", ""));
    }

    #[test]
    fn test_contains_short_needle() {
        assert!(simd_contains("abcdefgh", "abc"));
        assert!(!simd_contains("abcdefgh", "xyz"));
    }

    #[test]
    fn test_find_all_multiple() {
        let pos = simd_find_all("ababab", "ab");
        assert_eq!(pos, vec![0, 2, 4]);
    }

    #[test]
    fn test_find_all_none() {
        let pos = simd_find_all("hello", "xyz");
        assert!(pos.is_empty());
    }

    #[test]
    fn test_memmem() {
        assert_eq!(simd_memmem(b"hello world", b"world"), Some(6));
        assert_eq!(simd_memmem(b"hello", b"x"), None);
        assert_eq!(simd_memmem(b"hello", b""), Some(0));
    }
}
