#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import os
import hmac
import json
import hashlib
import base64
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

"""
PiscesL1 Watermark Protocol
- Unified payload schema (minimal, signed)
- HMAC-SHA256 signature
- Key discovery: configs['signing']['hmac_key'] or env PISCES_WM_HMAC_KEY
"""

def _canonicalize(obj: Dict[str, Any]) -> bytes:
    # Stable deterministic JSON for signing
    return json.dumps(obj, separators=(',', ':'), sort_keys=True, ensure_ascii=False).encode('utf-8')

def _load_signing_key(config: Dict[str, Any]) -> Tuple[bytes, str]:
    # Prefer config key, fallback to env
    key_id = (config.get('signing', {}) or {}).get('hmac_key_id') or 'default'
    key_str = (config.get('signing', {}) or {}).get('hmac_key') or os.getenv('PISCES_WM_HMAC_KEY')
    if not key_str:
        raise RuntimeError('Watermark signing key missing: set configs["signing"]["hmac_key"] or env PISCES_WM_HMAC_KEY')
    return key_str.encode('utf-8'), key_id

def create_payload(issuer: str,
                   model_id: str,
                   content_hash: str,
                   tenant: Optional[str] = None,
                   user_hash: Optional[str] = None,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "schema": "plx-wm-1.0",
        "issuer": issuer,
        "model_id": model_id,
        "gen_id": uuid.uuid4().hex[:16],
        "ts": datetime.utcnow().isoformat(),
        "tenant": tenant or None,
        "user_hash": user_hash or None,
        "content_hash": content_hash,
        "policy": {"disclosure": True}
    }
    if extra:
        # only include minimal whitelisted extras if needed
        payload["extra"] = extra
    return payload

def sign_payload(payload: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    key, key_id = _load_signing_key(config)
    msg = _canonicalize(payload)
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    sig_b64 = base64.b64encode(digest).decode('ascii')
    signed = dict(payload)
    signed["sig"] = sig_b64
    signed["sig_alg"] = "HMAC_SHA256"
    signed["key_id"] = key_id
    return signed

def verify_payload(signed_payload: Dict[str, Any], config: Dict[str, Any]) -> bool:
    try:
        sig_b64 = signed_payload.get("sig")
        if not sig_b64 or not signed_payload.get("sig_alg") == "HMAC_SHA256":
            return False
        # remove signature fields for canonicalization
        payload = dict(signed_payload)
        payload.pop("sig", None)
        payload.pop("sig_alg", None)
        payload.pop("key_id", None)
        key, _ = _load_signing_key(config)
        msg = _canonicalize(payload)
        expected = hmac.new(key, msg, hashlib.sha256).digest()
        actual = base64.b64decode(sig_b64.encode('ascii'))
        return hmac.compare_digest(expected, actual)
    except Exception:
        return False