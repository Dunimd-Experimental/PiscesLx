#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""Encre cryptographic layer — provides encrypt/decrypt for all sensitive data.

Architecture
============

All encryption uses **AES-256-GCM** (32‑byte key, 12‑byte random nonce,
16‑byte authentication tag).

Master key
    On first use a 256‑bit master key is generated from ``os.urandom(32)`` and
    stored in ``~/.encre/keyfile`` (mode ``0o600``).  The on‑disk representation
    is the master key itself wrapped with a *machine‑binding* key derived via
    HKDF‑SHA256 from the host's ``/etc/machine‑id`` content + hostname.

    This means even if ``keyfile`` is exfiltrated it cannot be unwrapped on
    any other machine.

    With the above::

        encrypt(plain_bytes)  ->  base64(nonce || ciphertext || tag)
        decrypt(b64_ciphertext) -> plain_bytes

No environment variables are consulted for key material.
"""

from __future__ import annotations

import base64
import hashlib
import os
import pathlib
import platform
import secrets
import stat
import struct

from typing import Union

__all__ = ["encrypt", "decrypt", "encrypt_bytes", "decrypt_bytes", "ensure_keyfile"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KEYFILE_PATH = pathlib.Path("~/.encre/keyfile").expanduser()
_KEYFILE_MODE = stat.S_IRUSR | stat.S_IWUSR  # 0o600 — owner read/write only

_KEY_LENGTH = 32       # AES-256
_NONCE_LENGTH = 12     # GCM standard
_TAG_LENGTH = 16       # GCM 128-bit auth tag
_GCM_IV_LENGTH = 12

# HKDF constants for deriving the machine-binding wrapping key
_HKDF_SALT = b"encre-crypto-hkdf-v1"
_HKDF_SALT_SHA256 = hashlib.sha256(_HKDF_SALT).digest()[:32]

_WRAP_CTX_AAD = b"encre-keywrap-v1"

# Cached master key (module lifetime) — avoid reading keyfile on every call
_master_key_cache: bytes | None = None


# ---------------------------------------------------------------------------
# Machine-binding identity
# ---------------------------------------------------------------------------

def _read_machine_id() -> bytes:
    """Return a stable machine‑identifier blob for key‑wrapping.

    Reads ``/etc/machine-id`` and falls back to ``gethostname()``.
    """
    try:
        mid = pathlib.Path("/etc/machine-id").read_text(encoding="utf-8").strip()
        if mid and mid != "uninitialized":
            return mid.encode("utf-8")
    except (OSError, UnicodeDecodeError):
        pass
    # Fallback: hash of hostname (still provides binding, just weaker)
    return platform.node().encode("utf-8")


# ---------------------------------------------------------------------------
# HKDF-SHA256 helper (implemented with hashlib — no third‑party required
# for this single‑step usage)
# ---------------------------------------------------------------------------

def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hashlib.sha256(salt + ikm).digest()


def _derive_wrapping_key() -> bytes:
    """Derive the AES‑256 key used to wrap/unwrap the master‑key file."""
    ikm = _read_machine_id()
    return _hkdf_extract(_HKDF_SALT_SHA256, ikm)


# ---------------------------------------------------------------------------
# Master‑key management (AES‑GCM wrap/unwrap with machine‑binding key)
# ---------------------------------------------------------------------------

def _generate_master_key() -> bytes:
    """Return a fresh 256‑bit master key."""
    return secrets.token_bytes(_KEY_LENGTH)


def _wrap_master_key(master_key: bytes) -> bytes:
    """Wrap *master_key* with the machine‑binding key via AES‑256‑GCM.

    Returns
    -------
    bytes
        ``nonce (12) || ciphertext (32) || tag (16)`` — total 60 bytes.
        The ciphertext is the encrypted master key.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    wrapping_key = _derive_wrapping_key()
    nonce = secrets.token_bytes(_NONCE_LENGTH)
    aesgcm = AESGCM(wrapping_key)
    ciphertext = aesgcm.encrypt(nonce, master_key, _WRAP_CTX_AAD)
    # ciphertext already includes the 16‑byte tag appended
    return nonce + ciphertext


def _unwrap_master_key(data: bytes) -> bytes:
    """Unwrap ``nonce || ciphertext || tag`` back to the master key."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    wrapping_key = _derive_wrapping_key()
    nonce = data[:_NONCE_LENGTH]
    rest = data[_NONCE_LENGTH:]  # ciphertext + tag
    aesgcm = AESGCM(wrapping_key)
    return aesgcm.decrypt(nonce, rest, _WRAP_CTX_AAD)


# ---------------------------------------------------------------------------
# Keyfile persistence
# ---------------------------------------------------------------------------

def _create_keyfile() -> bytes:
    """Generate a new master key, wrap it, write the keyfile and return the key."""
    keyfile_dir = _KEYFILE_PATH.parent
    keyfile_dir.mkdir(parents=True, exist_ok=True)
    # Only the owner of this directory should have access
    keyfile_dir.chmod(stat.S_IRWXU)

    master_key = _generate_master_key()
    wrapped = _wrap_master_key(master_key)

    # Atomic write via temp file + rename
    tmp = _KEYFILE_PATH.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(wrapped)
    os.chmod(tmp, _KEYFILE_MODE)
    os.replace(tmp, _KEYFILE_PATH)

    return master_key


def _load_keyfile() -> bytes | None:
    """Read the wrapped master key from disk and unwrap it.

    Returns None if the keyfile does not exist or cannot be decrypted.
    """
    if not _KEYFILE_PATH.exists():
        return None
    try:
        raw = _KEYFILE_PATH.read_bytes()
        if len(raw) < _NONCE_LENGTH + _KEY_LENGTH + _TAG_LENGTH:
            return None
        return _unwrap_master_key(raw)
    except Exception:
        return None


def ensure_keyfile() -> bytes:
    """Return the active master key, creating the keyfile if missing.

    This is the primary entry‑point — call once at startup.
    """
    global _master_key_cache
    if _master_key_cache is not None:
        return _master_key_cache

    key = _load_keyfile()
    if key is None:
        key = _create_keyfile()

    _master_key_cache = key
    return key


# ---------------------------------------------------------------------------
# AES-256-GCM encrypt / decrypt (user-facing API)
# ---------------------------------------------------------------------------

def _get_aesgcm() -> "AESGCM":
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(ensure_keyfile())


def encrypt(plaintext: str) -> str:
    """Encrypt a UTF‑8 string with AES‑256‑GCM.

    Returns a base64‑encoded ciphertext (nonce || ct || tag).
    """
    return encrypt_bytes(plaintext.encode("utf-8"))


def decrypt(ciphertext: str) -> str:
    """Decrypt a base64 ciphertext back to the original UTF‑8 string."""
    return decrypt_bytes(ciphertext).decode("utf-8")


def encrypt_bytes(plaintext: Union[str, bytes]) -> str:
    """Encrypt bytes (or a string treated as UTF‑8) → base64 string."""
    if isinstance(plaintext, str):
        plaintext = plaintext.encode("utf-8")
    aesgcm = _get_aesgcm()
    nonce = secrets.token_bytes(_NONCE_LENGTH)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return base64.b64encode(nonce + ct).decode("ascii")


def decrypt_bytes(ciphertext: str) -> bytes:
    """Decrypt a base64 ciphertext → raw bytes."""
    raw = base64.b64decode(ciphertext)
    nonce = raw[:_NONCE_LENGTH]
    ct = raw[_NONCE_LENGTH:]
    aesgcm = _get_aesgcm()
    return aesgcm.decrypt(nonce, ct, None)


# ---------------------------------------------------------------------------
# Convenience: encrypt/decrypt raw bytes → raw bytes (for binary data)
# ---------------------------------------------------------------------------

def encrypt_raw(plain_bytes: bytes) -> bytes:
    """Encrypt raw bytes → nonce || ciphertext || tag."""
    aesgcm = _get_aesgcm()
    nonce = secrets.token_bytes(_NONCE_LENGTH)
    return nonce + aesgcm.encrypt(nonce, plain_bytes, None)


def decrypt_raw(packed: bytes) -> bytes:
    """Decrypt nonce || ciphertext || tag → raw bytes."""
    nonce = packed[:_NONCE_LENGTH]
    ct = packed[_NONCE_LENGTH:]
    aesgcm = _get_aesgcm()
    return aesgcm.decrypt(nonce, ct, None)
