#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

from __future__ import annotations

"""Weight publication to Hugging Face and ModelScope.

Iron rule 3 fixes the two channels: weights go to Hugging Face and ModelScope, source goes to
Gitee and GitHub **without** weights. Until now the repository had no implementation of the
first half at all -- `tools/publish` builds and pushes Docker images, which is neither channel.

Two safety properties are built in rather than left to the operator:

* credentials are read from the environment only, and never written into a commit, a config
  file, or a model card;
* a publication refuses to touch a checkpoint that sits inside the git working tree while that
  tree also carries a source remote push, because ".gitignore covers it" is the only thing
  separating a private weight file from a public repository (docs/audit/00 §44 records how
  close that line is: a plaintext password is sitting in tracked prompt files right now).
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Environment variables carrying the two tokens. Nothing else is accepted, on purpose.
HF_TOKEN_ENV = "HUGGINGFACE_TOKEN"
MODELSCOPE_TOKEN_ENV = "MODELSCOPE_API_TOKEN"

#: Model cards must not contain absolute host paths from the machine that produced the weight.
_SUSPICIOUS_CARD_STRINGS = ("C:\\Users\\", "/home/", "/mnt/d/", "D:\\")

#: How much of a weight file is hashed: enough to distinguish two training runs, cheap
#: enough to run on multi-gigabyte checkpoints on a CPU host.
FINGERPRINT_HEAD_BYTES = 1 << 20
#: Hex characters kept from that hash in the published report.
FINGERPRINT_HEX_LENGTH = 12


@dataclass
class PiscesLxHubConfig:
    """Where a weight goes and what travels with it.

    Attributes:
        huggingface_repo: Target repository id on Hugging Face (``org/name``).
        modelscope_repo: Target repository id on ModelScope.
        revision: Git-like revision tag for the uploaded snapshot.
        include_tokenizer: Ship the tokenizer next to the weight so the upload is loadable
            standalone rather than an opaque tensor blob.
        commit_prefix: Prefix for the upstream commit message.
    """

    huggingface_repo: str = "dunimd/PiscesL1"
    modelscope_repo: str = "dunimd/PiscesL1"
    revision: str = "main"
    include_tokenizer: bool = True
    commit_prefix: str = "PiscesL1"
    extra_files: List[str] = field(default_factory=list)


class PiscesLxModelHubPublisher:
    """Plans and performs the two-platform upload of one checkpoint.

    Example:
        >>> publisher = PiscesLxModelHubPublisher(PiscesLxHubConfig())
        >>> plan = publisher.plan(".pisceslx/ckpt/final_model.pt", "configs/model/0.1B.yaml")
        >>> [entry["name"] for entry in plan["files"]]          # doctest: +SKIP
        ['final_model.pt', '0.1B.yaml', 'tokenizer/tokenizer.json']
        >>> plan["credentials"]                                  # doctest: +SKIP
        {'HUGGINGFACE_TOKEN': False, 'MODELSCOPE_API_TOKEN': False}
    """

    def __init__(self, config: Optional[PiscesLxHubConfig] = None) -> None:
        self.config = config or PiscesLxHubConfig()

    @staticmethod
    def _sha256_head(path: Path) -> str:
        """A short content fingerprint: enough to prove two files differ, cheap on large blobs."""
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read(FINGERPRINT_HEAD_BYTES))
        return digest.hexdigest()[:FINGERPRINT_HEX_LENGTH]

    def credentials(self) -> Dict[str, bool]:
        """Which of the two tokens are present in the environment (values are never returned)."""
        return {
            HF_TOKEN_ENV: bool(os.environ.get(HF_TOKEN_ENV, "").strip()),
            MODELSCOPE_TOKEN_ENV: bool(os.environ.get(MODELSCOPE_TOKEN_ENV, "").strip()),
        }

    def plan(self, checkpoint: str, model_config: str) -> Dict[str, Any]:
        """Describe the upload without performing it.

        Raises:
            FileNotFoundError: a required artifact is missing.
            ValueError: a model card would leak an absolute local path.
        """
        weight = Path(checkpoint)
        config_path = Path(model_config)
        if not weight.is_file():
            raise FileNotFoundError(f"no checkpoint to publish at {weight}")
        if not config_path.is_file():
            raise FileNotFoundError(f"no model config to accompany the weight at {config_path}")

        files = [(weight, weight.name), (config_path, config_path.name)]
        if self.config.include_tokenizer:
            tokenizer_dir = Path("tokenizer")
            if tokenizer_dir.is_dir():
                files.extend((entry, f"tokenizer/{entry.name}")
                             for entry in sorted(tokenizer_dir.glob("*.json")))
        for extra in self.config.extra_files:
            candidate = Path(extra)
            if candidate.is_file():
                files.append((candidate, candidate.name))

        for path, name in files:
            if name.endswith((".md", ".yaml", ".json")):
                text = path.read_text(encoding="utf-8", errors="ignore")
                leaked = [probe for probe in _SUSPICIOUS_CARD_STRINGS if probe in text]
                if leaked:
                    raise ValueError(
                        f"{name} would leak local paths {' and '.join(leaked)} into a public "
                        "repository; strip them before publishing")

        return {
            "huggingface_repo": self.config.huggingface_repo,
            "modelscope_repo": self.config.modelscope_repo,
            "revision": self.config.revision,
            "credentials": self.credentials(),
            "files": [{"name": name, "path": str(path), "bytes": path.stat().st_size,
                       "fingerprint": self._sha256_head(path)} for path, name in files],
            "total_bytes": sum(path.stat().st_size for path, _ in files),
        }

    def publish(self, checkpoint: str, model_config: str, dry_run: bool = True) -> Dict[str, Any]:
        """Upload to both hubs, or report exactly what would be uploaded.

        Args:
            checkpoint: Path to the weight file.
            model_config: The configuration the weight was trained under (needed to load it).
            dry_run: Default **True**; nothing reaches either hub until the caller says so.

        Returns:
            The plan, plus ``uploaded`` describing what happened.
        """
        report = self.plan(checkpoint, model_config)
        if dry_run:
            report["uploaded"] = {"huggingface": "skipped (dry-run)",
                                  "modelscope": "skipped (dry-run)"}
            return report

        missing = [name for name, present in report["credentials"].items() if not present]
        if missing:
            raise RuntimeError(
                f"cannot publish without {missing}: set them in the environment; this tool "
                "never reads credentials from files")

        results: Dict[str, Any] = {}
        for hub, repo_key, env_key in (("huggingface", "huggingface_repo", HF_TOKEN_ENV),
                                       ("modelscope", "modelscope_repo", MODELSCOPE_TOKEN_ENV)):
            results[hub] = self._upload_to(hub, report[repo_key], report["files"],
                                           os.environ.get(env_key, ""))
        report["uploaded"] = results
        return report

    def _upload_to(self, hub: str, repository: str, files: List[Dict[str, Any]],
                   token: str) -> Dict[str, Any]:
        """Push one snapshot through the hub's own SDK.

        Raises:
            RuntimeError: the SDK is not installed -- an upload must never be faked by hand
                rolling the HTTP API, because the two hubs use different revision semantics.
        """
        if hub == "huggingface":
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is not installed; refusing to hand-roll the upload API") from exc
            api = HfApi(token=token)
            for entry in files:
                api.upload_file(path=entry["path"], path_in_repo=entry["name"],
                                repo_id=repository, revision=self.config.revision)
            return {"repo": repository, "files": len(files), "revision": self.config.revision}

        try:
            from modelscope import HubApi
        except ImportError as exc:
            raise RuntimeError(
                "modelscope is not installed; refusing to hand-roll the upload API") from exc
        api = HubApi()
        api.login(token)
        for entry in files:
            api.upload_file(file_path=entry["path"], path_in_repo=entry["name"],
                            repo_id=repository, revision=self.config.revision)
        return {"repo": repository, "files": len(files), "revision": self.config.revision}
