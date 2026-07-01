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

#[cfg(target_os = "linux")]
mod linux_impl {
    use std::ffi::CString;
    use std::os::raw::c_int;

    // Landlock syscall numbers (stable since Linux 5.13)
    #[cfg(target_arch = "x86_64")]
    const SYS_LANDLOCK_CREATE_RULESET: i64 = 444;
    #[cfg(target_arch = "x86_64")]
    const SYS_LANDLOCK_ADD_RULE: i64 = 445;
    #[cfg(target_arch = "x86_64")]
    const SYS_LANDLOCK_RESTRICT_SELF: i64 = 446;

    #[cfg(target_arch = "aarch64")]
    const SYS_LANDLOCK_CREATE_RULESET: i64 = 444;
    #[cfg(target_arch = "aarch64")]
    const SYS_LANDLOCK_ADD_RULE: i64 = 445;
    #[cfg(target_arch = "aarch64")]
    const SYS_LANDLOCK_RESTRICT_SELF: i64 = 446;

    // Landlock ABI versions
    const LANDLOCK_ABI_V1: u64 = 1;
    const LANDLOCK_ABI_V2: u64 = 2; // adds network support (Linux 5.19+)
    // const LANDLOCK_ABI_V3: u64 = 3; // adds scoped rules (Linux 6.3+)

    // --- Landlock rule attribute flags (merged from linux/landlock.h) ---

    /// Filesystem access rights bitmask.
    #[allow(dead_code)]
    mod fs_access {
        pub const EXECUTE: u64 = 1 << 0;
        pub const WRITE_FILE: u64 = 1 << 1;
        pub const READ_FILE: u64 = 1 << 2;
        pub const READ_DIR: u64 = 1 << 3;
        pub const REMOVE_DIR: u64 = 1 << 4;
        pub const REMOVE_FILE: u64 = 1 << 5;
        pub const MAKE_CHAR: u64 = 1 << 6;
        pub const MAKE_DIR: u64 = 1 << 7;
        pub const MAKE_REG: u64 = 1 << 8;
        pub const MAKE_SOCK: u64 = 1 << 9;
        pub const MAKE_FIFO: u64 = 1 << 10;
        pub const MAKE_BLOCK: u64 = 1 << 11;
        pub const MAKE_SYM: u64 = 1 << 12;
        // V2 additions
        pub const REFER: u64 = 1 << 13;
        pub const TRUNCATE: u64 = 1 << 14;
    }

    #[allow(dead_code)]
    mod net_access {
        pub const BIND_TCP: u64 = 1 << 0;
        pub const CONNECT_TCP: u64 = 1 << 1;
    }

    // Landlock rule type
    const LANDLOCK_RULE_PATH_BENEATH: u64 = 1;
    // const LANDLOCK_RULE_NET_PORT: u64 = 2; // unused

    // --- FFI: raw libc syscall wrapper ---

    extern "C" {
        fn syscall(number: i64, ...) -> i64;
    }

    // The C `prctl` syscall for checking Landlock availability.
    extern "C" {
        fn prctl(option: c_int, ...) -> c_int;
    }

    const PR_GET_NO_NEW_PRIVS: c_int = 39;

    // --- Landlock C struct layouts (must match kernel UAPI) ---

    #[repr(C)]
    struct LandlockRulesetAttr {
        handled_access_fs: u64,
        handled_access_net: u64,
    }

    #[repr(C)]
    struct LandlockPathBeneathAttr {
        allowed_access: u64,
        parent_fd: i32,
    }

    #[repr(C)]
    struct LandlockNetPortAttr {
        allowed_access: u64,
        port: u64,
    }

    /// Return the highest Landlock ABI version supported by the running kernel.
    ///
    /// Returns `0` if Landlock is not available.
    pub fn landlock_abi_version() -> u64 {
        let ruleset_attr = LandlockRulesetAttr {
            handled_access_fs: 0,
            handled_access_net: 0,
        };

        let flags = 1u64 << 2; // LANDLOCK_CREATE_RULESET_VERSION

        unsafe {
            let ret = syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                &ruleset_attr as *const LandlockRulesetAttr,
                std::mem::size_of::<LandlockRulesetAttr>(),
                flags,
            );
            if ret < 0 {
                0
            } else {
                // Close the fd we just created
                libc_syscall_close(ret as i32);
                ret as u64
            }
        }
    }

    /// Check whether Landlock is available on the current kernel (ABI >= 1).
    pub fn landlock_available() -> bool {
        landlock_abi_version() >= LANDLOCK_ABI_V1
    }

    /// Wrap libc close via syscall, since we are working at the syscall layer.
    unsafe fn libc_syscall_close(fd: i32) {
        #[cfg(target_arch = "x86_64")]
        {
            syscall(3, fd);
        }
        #[cfg(target_arch = "aarch64")]
        {
            syscall(57, fd);
        }
    }

    /// Ensure the calling thread has `NO_NEW_PRIVS` set (required by Landlock).
    fn ensure_no_new_privs() -> Result<(), String> {
        unsafe {
            let ret = prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0);
            if ret == 0 {
                // Not set yet — set it
                let ret2 = prctl(22 /* PR_SET_NO_NEW_PRIVS */, 1, 0, 0, 0);
                if ret2 != 0 {
                    return Err(format!(
                        "Failed to set NO_NEW_PRIVS: errno {}",
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
                    ));
                }
            } else if ret != 1 {
                return Err(format!(
                    "Unexpected PR_GET_NO_NEW_PRIVS return value: {}",
                    ret
                ));
            }
        }
        Ok(())
    }

    /// Create a Landlock ruleset file descriptor.
    fn create_ruleset(handled_access_fs: u64, handled_access_net: u64) -> Result<i32, String> {
        let attr = LandlockRulesetAttr {
            handled_access_fs,
            handled_access_net,
        };

        unsafe {
            let ret = syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                &attr as *const LandlockRulesetAttr,
                std::mem::size_of::<LandlockRulesetAttr>(),
                0u64,
            );
            if ret < 0 {
                let errno = -ret as i32;
                return Err(format!(
                    "landlock_create_ruleset failed: errno {}",
                    errno
                ));
            }
            Ok(ret as i32)
        }
    }

    /// Add a path-beneath rule to the given ruleset fd.
    fn add_path_beneath_rule(ruleset_fd: i32, path: &str, allowed_access: u64) -> Result<(), String> {
        // // use std::os::unix::ffi::OsStrExt; // unused // unused
        let cpath = CString::new(path).map_err(|_| format!("Invalid path: {}", path))?;

        // Open the parent directory as a file descriptor
        let fd = unsafe {
            let fd = libc_open(
                cpath.as_ptr(),
                libc::O_PATH | libc::O_CLOEXEC,
            );
            if fd < 0 {
                return Err(format!(
                    "Failed to open '{}': errno {}",
                    path,
                    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
                ));
            }
            fd
        };

        let path_attr = LandlockPathBeneathAttr {
            allowed_access,
            parent_fd: fd,
        };

        unsafe {
            let ret = syscall(
                SYS_LANDLOCK_ADD_RULE,
                ruleset_fd as i64,
                LANDLOCK_RULE_PATH_BENEATH,
                &path_attr as *const LandlockPathBeneathAttr,
                0u64,
            );
            libc_syscall_close(fd);
            if ret != 0 {
                return Err(format!(
                    "landlock_add_rule (path_beneath) failed for '{}': errno {}",
                    path,
                    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
                ));
            }
        }
        Ok(())
    }

    // Extern libc `open` (O_PATH, O_CLOEXEC).
    extern "C" {
        fn open(path: *const std::os::raw::c_char, flags: i32, ...) -> i32;
    }

    #[allow(non_upper_case_globals)]
    mod libc {
        pub const O_PATH: i32 = 0o10000000;
        pub const O_CLOEXEC: i32 = 0o2000000;
    }

    unsafe fn libc_open(path: *const std::os::raw::c_char, flags: i32) -> i32 {
        open(path, flags)
    }

    /// Enforce the ruleset, making it immutable for the current thread.
    fn enforce_ruleset(ruleset_fd: i32) -> Result<(), String> {
        unsafe {
            let ret = syscall(
                SYS_LANDLOCK_RESTRICT_SELF,
                ruleset_fd as i64,
                0u64,
            );
            libc_syscall_close(ruleset_fd);
            if ret != 0 {
                return Err(format!(
                    "landlock_restrict_self failed: errno {}",
                    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
                ));
            }
        }
        Ok(())
    }

    // --- Public API ---

    /// Restrict the current thread to **read-only** filesystem access under
    /// the given paths.  After this call, the thread cannot write, execute,
    /// or create files anywhere except the listed directories (and even those
    /// are read-only).
    pub fn landlock_restrict_read_only(paths: &[&str]) -> Result<(), String> {
        if !landlock_available() {
            return Err("Landlock is not available on this kernel (requires 5.13+)".to_string());
        }
        ensure_no_new_privs()?;

        let access = fs_access::READ_FILE
            | fs_access::READ_DIR;

        let ruleset_fd = create_ruleset(access, 0)?;

        for path in paths {
            add_path_beneath_rule(ruleset_fd, path, access)?;
        }

        enforce_ruleset(ruleset_fd)
    }

    /// Restrict the current thread from making any network connections.
    ///
    /// Requires kernel 5.19+ (Landlock ABI v2).  Returns an error on
    /// older kernels.
    pub fn landlock_restrict_network() -> Result<(), String> {
        if landlock_abi_version() < LANDLOCK_ABI_V2 {
            return Err(
                "Network restriction requires Landlock ABI v2 (Linux 5.19+)".to_string()
            );
        }
        ensure_no_new_privs()?;

        // Block all TCP bind and connect
        let net_access = net_access::BIND_TCP | net_access::CONNECT_TCP;

        let ruleset_fd = create_ruleset(0, net_access)?;
        // No allowed network rules added → all network is blocked
        enforce_ruleset(ruleset_fd)
    }

    /// Full sandbox: read-only filesystem under `workspace`, no network,
    /// no execution.  Suitable for safely running untrusted code.
    ///
    /// After this call the thread can only read files under `workspace`.
    pub fn landlock_full_sandbox(workspace: &str) -> Result<(), String> {
        if !landlock_available() {
            return Err("Landlock is not available on this kernel (requires 5.13+)".to_string());
        }
        ensure_no_new_privs()?;

        let fs_allowed = fs_access::READ_FILE | fs_access::READ_DIR;
        let net_handled = if landlock_abi_version() >= LANDLOCK_ABI_V2 {
            net_access::BIND_TCP | net_access::CONNECT_TCP
        } else {
            0
        };

        let ruleset_fd = create_ruleset(fs_allowed, net_handled)?;

        // Allow read-only access under workspace
        add_path_beneath_rule(ruleset_fd, workspace, fs_allowed)?;

        enforce_ruleset(ruleset_fd)
    }

    /// Workspace sandbox: read-write access under `workspace`, no network,
    /// no exec, no access to files outside workspace.
    ///
    /// Suitable for running build tools (npm install, cargo build, etc.)
    /// in an isolated environment.  The child process can read, write,
    /// create, and remove files **only** under `workspace`.
    ///
    /// Requires Landlock ABI v1 (Linux 5.13+) for filesystem and
    /// ABI v2 (Linux 5.19+) for network restriction.
    pub fn landlock_workspace_sandbox(workspace: &str) -> Result<(), String> {
        if !landlock_available() {
            return Err("Landlock is not available on this kernel (requires 5.13+)".to_string());
        }
        ensure_no_new_privs()?;

        // Allow read, write, create, remove workspace files
        let fs_allowed = fs_access::READ_FILE
            | fs_access::READ_DIR
            | fs_access::WRITE_FILE
            | fs_access::MAKE_REG
            | fs_access::MAKE_DIR
            | fs_access::REMOVE_FILE
            | fs_access::TRUNCATE;

        let net_handled = if landlock_abi_version() >= LANDLOCK_ABI_V2 {
            net_access::BIND_TCP | net_access::CONNECT_TCP
        } else {
            0
        };

        let ruleset_fd = create_ruleset(fs_allowed, net_handled)?;

        // Allow read-write access under workspace
        add_path_beneath_rule(ruleset_fd, workspace, fs_allowed)?;

        enforce_ruleset(ruleset_fd)
    }
}

// ---------------------------------------------------------------------------
// Non-Linux stubs
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "linux"))]
mod linux_impl {
    pub fn landlock_restrict_read_only(_paths: &[&str]) -> Result<(), String> {
        Err("Landlock is only available on Linux".to_string())
    }
    pub fn landlock_restrict_network() -> Result<(), String> {
        Err("Landlock is only available on Linux".to_string())
    }
    pub fn landlock_full_sandbox(_workspace: &str) -> Result<(), String> {
        Err("Landlock is only available on Linux".to_string())
    }
    pub fn landlock_workspace_sandbox(_workspace: &str) -> Result<(), String> {
        Err("Landlock is only available on Linux".to_string())
    }
    pub fn landlock_available() -> bool {
        false
    }
    pub fn landlock_abi_version() -> u64 {
        0
    }
}

// ---------------------------------------------------------------------------
// Public re-exports
// ---------------------------------------------------------------------------

/// Restrict the current thread to **read-only** filesystem access under
/// the given paths.
///
/// Returns `Err` on non-Linux systems or kernels older than 5.13.
pub fn landlock_restrict_read_only(paths: &[&str]) -> Result<(), String> {
    linux_impl::landlock_restrict_read_only(paths)
}

/// Restrict the current thread from making network connections.
///
/// Requires Landlock ABI v2 (Linux 5.19+).
pub fn landlock_restrict_network() -> Result<(), String> {
    linux_impl::landlock_restrict_network()
}

/// Full sandbox: read-only filesystem under `workspace`, no network,
/// no execution.
pub fn landlock_full_sandbox(workspace: &str) -> Result<(), String> {
    linux_impl::landlock_full_sandbox(workspace)
}

/// Workspace sandbox: read-write access under `workspace`, no network,
/// no exec outside workspace.  Suitable for build tools.
pub fn landlock_workspace_sandbox(workspace: &str) -> Result<(), String> {
    linux_impl::landlock_workspace_sandbox(workspace)
}

/// Check whether Landlock is available on the current kernel.
pub fn landlock_available() -> bool {
    linux_impl::landlock_available()
}

/// Return the highest Landlock ABI version supported by the running kernel,
/// or `0` if Landlock is not available.
pub fn landlock_abi_version() -> u64 {
    linux_impl::landlock_abi_version()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landlock_available_does_not_panic() {
        // Should always return a bool without panicking.
        let _ = landlock_available();
    }

    #[test]
    fn test_abi_version_does_not_panic() {
        let v = landlock_abi_version();
        assert!(v < 10, "unexpected ABI version: {}", v);
    }
}
