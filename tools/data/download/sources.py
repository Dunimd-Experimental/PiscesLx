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

import os
import urllib.request
import urllib.error
import sys
import json
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.dc import PiscesLxLogger
from typing import Any, Dict, List, Optional

from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Data", file_path=get_log_file("PiscesLx.Tools.Data"), enable_file=True)

# Verbose switch: set PISCESLX_DOWNLOAD_VERBOSE=1 to see detailed debug logs
_VERBOSE = (os.getenv("PISCESLX_DOWNLOAD_VERBOSE", "0") == "1")

HF_MIRROR_URL = "https://hf-mirror.com"


def _classify_modelscope_import_error(exc: BaseException, traceback_text: str) -> str:
    """
    Classify the most common reasons `modelscope.msdatasets` fails to import.

    Args:
        exc: The exception instance caught at the import boundary.
        traceback_text: The string form of the traceback, used for keyword
            matching against known broken-call chains.

    Returns:
        A short human-readable description of the most likely root cause.
    """
    msg = f"{type(exc).__name__}: {exc}".lower()
    tb = traceback_text.lower()

    # transformers ≥ 4.46 stopped re-exporting PreTrainedModel from
    # `transformers` directly. ModelScope ≥ 1.18 imports it the old way, so
    # the error string is "could not import module 'PreTrainedModel'".
    if "pretrainedmodel" in msg or "pretrainedmodel" in tb:
        return (
            "transformers ≥ 4.46 removed the top-level `PreTrainedModel` "
            "symbol that older `modelscope` releases still import. Either "
            "upgrade `modelscope` to a version that supports current "
            "transformers, or pin `transformers<4.46`."
        )

    # Pinned transformers but the actual broken import is somewhere else
    # in modelscope's transitive imports (datasets, einops, timm, ...).
    if "importerror" in msg or "modulenotfounderror" in msg:
        # Pull the offending module name out of the exception if we can.
        missing = getattr(exc, "name", None) or ""
        if missing:
            return f"a transitive dependency `{missing}` is missing or version-incompatible"
        return "a transitive dependency is missing or version-incompatible"

    if "syntaxerror" in msg:
        return "a Python syntax error was raised while importing modelscope (likely a version mismatch)"

    return "an unknown import-time error in the modelscope stack"


class PiscesLxToolsDataSourceRouter:
    """
    A router class responsible for loading datasets from different sources.
    """
    
    @staticmethod
    def check_huggingface_connectivity() -> bool:
        """
        Check if HuggingFace is accessible.
        
        Returns:
            bool: True if HuggingFace is accessible, False otherwise.
        """
        try:
            # Try to access HuggingFace hub
            response = urllib.request.urlopen('https://huggingface.co', timeout=5)
            return True
        except Exception as e:
            if _VERBOSE:
                _LOG.debug(f"HuggingFace connectivity check failed: {e}")
            return False

    @staticmethod
    def setup_hf_mirror(verbose: bool = True) -> None:
        """
        Set up HuggingFace mirror if the main site is not accessible.
        """
        # Set environment variable for HuggingFace mirror
        os.environ['HF_ENDPOINT'] = HF_MIRROR_URL
        # Also set for datasets library compatibility
        os.environ['HUGGINGFACE_HUB_ENDPOINT'] = HF_MIRROR_URL
        if verbose:
            _LOG.info("Using HuggingFace mirror: " + HF_MIRROR_URL)

    def __init__(self):
        """Initialize the source router. Log initialization if verbose mode is enabled."""
        # Mirror is configured lazily by each source loader so we don't pollute
        # the log when the user has chosen a non-HF source.
        if _VERBOSE:
            _LOG.debug("SourceRouter initialized")
        # Last error captured during a `load()` call, exposed for the caller
        # (e.g. download_worker) so it can be reported in the resume state
        # without re-running the load.
        self.last_error: Optional[str] = None

    def load(self, dataset_name: str, kwargs: Dict[str, Any] = None,
             preferred_sources: List[str] = None, **extra_kwargs) -> Optional[Any]:
        """
        Attempt to load a dataset from the specified sources in order of preference.

        Args:
            dataset_name (str): Name of the dataset to load.
            kwargs (Dict[str, Any], optional): Additional arguments for dataset loading. Defaults to None.
            preferred_sources (List[str], optional): List of preferred sources in order.
                Defaults to ["modelscope", "huggingface"].

        Returns:
            Optional[Any]: The loaded dataset object if successful, None otherwise.

        Note:
            The router exposes ``self.last_error`` describing why the most
            recent source failed, so callers can persist it.
        """
        if kwargs is None:
            kwargs = {}
        # Merge extra_kwargs into kwargs
        kwargs.update(extra_kwargs)
        if preferred_sources is None:
            preferred_sources = ["modelscope", "huggingface"]

        self.last_error = None
        failures: List[str] = []

        # Try each source in order of preference
        for idx, source in enumerate(preferred_sources):
            try:
                if source == "modelscope":
                    _LOG.info(f"[source={source}] trying ModelScope for {dataset_name} (kwargs={kwargs})")
                    result = self._load_from_modelscope(dataset_name, kwargs)
                    if result is not None:
                        return result
                    msg = f"ModelScope returned no data for {dataset_name}"
                    _LOG.warning(f"[source={source}] {msg}")
                    failures.append(f"{source}: {msg}")
                elif source == "huggingface":
                    _LOG.info(f"[source={source}] trying HuggingFace for {dataset_name} (kwargs={kwargs})")
                    result = self._load_from_huggingface(dataset_name, kwargs)
                    if result is not None:
                        return result
                    msg = f"HuggingFace returned no data for {dataset_name}"
                    _LOG.warning(f"[source={source}] {msg}")
                    failures.append(f"{source}: {msg}")
                else:
                    msg = f"unknown source '{source}' for {dataset_name}, skipping"
                    _LOG.warning(f"[source={source}] {msg}")
                    failures.append(f"{source}: {msg}")
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                _LOG.error(f"[source={source}] exception while loading {dataset_name}: {msg}")
                if _VERBOSE:
                    _LOG.debug(f"Router load error: source={source} dataset={dataset_name} kwargs={kwargs}: {e}")
                failures.append(f"{source}: {msg}")
                continue

        self.last_error = "; ".join(failures) if failures else "no source was attempted"
        _LOG.error(
            f"All configured sources {preferred_sources} failed for {dataset_name}: "
            f"{self.last_error}"
        )
        return None

    def _load_from_modelscope(self, dataset_name: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """
        Load a dataset from the ModelScope platform.

        Args:
            dataset_name (str): Name of the dataset to load.
            kwargs (Dict[str, Any]): Additional arguments for dataset loading.

        Returns:
            Optional[Any]: The loaded dataset object if successful, None otherwise.
        """
        import traceback

        # 1. Probe: is the `modelscope` package itself importable? This is the
        #    "not installed at all" branch.
        try:
            import modelscope  # type: ignore  # noqa: F401
        except Exception as e:
            _LOG.error(
                "ModelScope package is not installed: "
                f"{type(e).__name__}: {e}. "
                "Install it with `pip install -U modelscope`, or change the "
                "dataset's `source` to another provider."
            )
            return None

        # 2. Probe: does `modelscope.msdatasets.MsDataset` import cleanly?
        #    `modelscope` itself can be installed while this submodule is
        #    broken because it drags in `transformers.modeling_utils` which
        #    references symbols that have moved between transformers versions
        #    (notably `PreTrainedModel`, which is no longer re-exported from
        #    the top-level `transformers` namespace in newer releases).
        try:
            from modelscope.msdatasets import MsDataset  # type: ignore
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            root_cause = _classify_modelscope_import_error(e, tb)
            _LOG.error(
                "Failed to import `modelscope.msdatasets.MsDataset`: "
                f"{type(e).__name__}: {e}. "
                f"Likely cause: {root_cause}. "
                "Try `pip install -U modelscope transformers`, or pin "
                "`transformers<4.46` if your `modelscope` release does not "
                "support newer transformers yet."
            )
            return None

        # 3. Real load. Surface the original traceback tail if anything blows
        #    up here so the caller can see the actual failure point.
        try:
            result = MsDataset.load(dataset_name, **kwargs)
            _LOG.info(f"Successfully loaded dataset {dataset_name} from ModelScope")
            return result
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            _LOG.error(
                f"ModelScope load failed for {dataset_name} with kwargs={kwargs}: "
                f"{type(e).__name__}: {e}"
            )
            if _VERBOSE:
                _LOG.debug(f"ModelScope load traceback:\n{tb}")
            return None

    def _load_from_huggingface(self, dataset_name: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """
        Load a dataset from the HuggingFace platform.

        Args:
            dataset_name (str): Name of the dataset to load.
            kwargs (Dict[str, Any]): Additional arguments for dataset loading.

        Returns:
            Optional[Any]: The loaded dataset object if successful, None otherwise.
        """
        try:
            # Ensure mirror is set up before loading
            PiscesLxToolsDataSourceRouter.setup_hf_mirror()
            from datasets import load_dataset
            _LOG.info(f"Attempting to load dataset {dataset_name} from HuggingFace with kwargs={kwargs}")
            result = load_dataset(dataset_name, **kwargs)
            _LOG.info(f"Successfully loaded dataset {dataset_name} from HuggingFace")
            return result
        except Exception as e:
            _LOG.error(f"HuggingFace load failed for {dataset_name} with kwargs={kwargs}: {e}")
            if _VERBOSE:
                _LOG.debug(f"HuggingFace load failed for {dataset_name} with kwargs={kwargs}: {e}")
            return None

    @staticmethod
    def detect_available_splits(dataset_name: str, source: str | None = None) -> list[str]:
        """
        Detect available splits for a dataset on a specific source without brute-force attempts.

        Args:
            dataset_name (str): The name of the dataset repository.
            source (str | None, optional): The source platform, either "modelscope" or "huggingface". 
                If None, defaults to "modelscope".

        Returns:
            list[str]: A list of available split names. If empty, the caller should try direct load without a split.
                If only direct load works, returns ["__direct__"].
        """
        # List of common split names to probe
        candidates = [
            "train", "train_full", "train_all",
            "validation", "valid", "dev",
            "test", "eval", "test_all",
        ]

        src = (source or "modelscope").strip().lower()
        available: list[str] = []

        # Try each candidate split
        for split in candidates:
            try:
                if src == "modelscope":
                    from modelscope.msdatasets import MsDataset  # type: ignore
                    _ = MsDataset.load(dataset_name, split=split)
                elif src == "huggingface":
                    from datasets import load_dataset  # type: ignore
                    _ = load_dataset(dataset_name, split=split)
                else:
                    continue
                available.append(split)
            except Exception as e:
                if _VERBOSE:
                    _LOG.debug(f"Split probe failed: source={src} dataset={dataset_name} split={split}: {e}")
                continue

        if not available:
            # Try to load the dataset directly without specifying a split
            try:
                if src == "modelscope":
                    from modelscope.msdatasets import MsDataset  # type: ignore
                    _ = MsDataset.load(dataset_name)
                elif src == "huggingface":
                    from datasets import load_dataset  # type: ignore
                    _ = load_dataset(dataset_name)
                available.append("__direct__")
            except Exception as e:
                # No available splits or direct load
                if _VERBOSE:
                    _LOG.debug(f"Direct probe failed: source={src} dataset={dataset_name}: {e}")

        return available

    @staticmethod
    def to_hf_if_needed(ds: Any) -> Any:
        """
        Convert a dataset to HuggingFace format if necessary.

        - If the dataset is already a HuggingFace dataset (has save_to_disk method), return it as-is.
        - If the dataset is a ModelScope dataset with to_hf_dataset method, convert it and return.
        - Otherwise, return the original object.

        Args:
            ds (Any): The dataset object to potentially convert.

        Returns:
            Any: The original or converted dataset object.
        """
        try:
            # Check if it's already a HuggingFace dataset
            if hasattr(ds, "save_to_disk"):
                return ds
            # Try to convert if it's a ModelScope dataset
            if hasattr(ds, "to_hf_dataset"):
                try:
                    return ds.to_hf_dataset()  # type: ignore[attr-defined]
                except Exception as e:
                    if _VERBOSE:
                        _LOG.debug(f"to_hf_if_needed conversion failed: {e}")
        except Exception as e:
            if _VERBOSE:
                _LOG.debug(f"to_hf_if_needed conversion failed: {e}")
        return ds
