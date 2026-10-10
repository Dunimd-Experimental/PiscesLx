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

"""``train_launch`` tool -- launch a training run as a detached background
subprocess (``manage.py train ...``) and track it for later monitoring.

The heavy training process runs in a child process so the agent event loop is
never blocked.  Two module-level helpers are exposed so other training tools
(``train_orchestrate``) can reuse the same argv building / launch logic:

* :func:`build_train_argv` -- map a parameter dict to a ``manage.py train`` argv list.
* :func:`launch_train`     -- spawn the process and return a ``run_id``/pid/log path.

Only the EnTA-trainable flags are surfaced; the full flag set lives in
``manage.py``.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
import uuid
from typing import Any

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

logger = PiscesLxLogger(
    "EnTA.Tools.TrainLaunch",
    file_path=get_log_file("EnTA.Tools.TrainLaunch"),
    enable_file=True,
)

from enta.tools.base import build_tool

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[5]
MANAGE_PY = PROJECT_ROOT / "manage.py"
RUNS_ROOT = PROJECT_ROOT / "logs" / "train_runs"

_VALID_MODES = ("standard", "distill", "rlhf", "enta")


def build_train_argv(
    *,
    mode: str = "standard",
    train_config: str | None = None,
    model_size: str | None = None,
    dataset: str | None = None,
    resume_ckpt: str | None = None,
    enta_config: str | None = None,
    distill: bool = False,
    rlhf: bool = False,
    phases: list[str] | None = None,
    max_iterations: int | None = None,
    extra_flags: dict[str, Any] | None = None,
) -> list[str]:
    """Build the ``manage.py train`` argument vector from a parameter dict.

    Mirrors the flag mapping in ``manage.py`` (``train`` sub-command).  Unknown
    options can be passed through ``extra_flags`` (``bool`` -> flag-only,
    everything else -> ``--key value``).
    """
    argv: list[str] = [sys.executable, str(MANAGE_PY), "train"]

    if mode == "enta":
        argv.append("--enta")
    if enta_config:
        argv += ["--enta_config", str(enta_config)]
    if train_config:
        argv += ["--train_config", str(train_config)]
    if model_size:
        argv += ["--model_size", str(model_size)]
    if dataset:
        argv += ["--dataset", str(dataset)]
    if resume_ckpt:
        argv += ["--resume_ckpt", str(resume_ckpt)]
    if distill:
        argv.append("--distill")
    if rlhf:
        argv.append("--rlhf")
    if phases:
        argv += ["--phases", ",".join(str(p) for p in phases)]
    if max_iterations is not None:
        argv += ["--max_iterations", str(max_iterations)]

    if extra_flags:
        for key, value in extra_flags.items():
            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            elif value is not None:
                argv += [flag, str(value)]

    return argv


def launch_train(argv: list[str], *, run_dir: pathlib.Path | None = None) -> dict[str, Any]:
    """Spawn the training process in the background and return run metadata.

    The child inherits the current environment and runs with ``cwd`` set to the
    project root so that ``manage.py`` can resolve its own imports.  Its merged
    stdout/stderr is streamed to ``<run_dir>/train.log``.
    """
    run_id = f"train_{uuid.uuid4().hex[:10]}"
    run_dir = run_dir or (RUNS_ROOT / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        argv,
        cwd=str(PROJECT_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    info = {
        "run_id": run_id,
        "pid": proc.pid,
        "log_path": str(log_path),
        "run_dir": str(run_dir),
        "argv": argv,
        "start_time": time.time(),
    }
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Launched training %s (pid=%s): %s", run_id, proc.pid, " ".join(argv))
    return info


async def _train_launch_execute(**kwargs: Any) -> dict[str, Any]:
    mode = kwargs.get("mode", "standard")
    if mode not in _VALID_MODES:
        return {
            "success": False,
            "error": f"Unknown mode '{mode}'. Expected one of {_VALID_MODES}.",
        }

    try:
        argv = build_train_argv(
            mode=mode,
            train_config=kwargs.get("train_config"),
            model_size=kwargs.get("model_size"),
            dataset=kwargs.get("dataset"),
            resume_ckpt=kwargs.get("resume_ckpt"),
            enta_config=kwargs.get("enta_config"),
            distill=bool(kwargs.get("distill", False)),
            rlhf=bool(kwargs.get("rlhf", False)),
            phases=kwargs.get("phases"),
            max_iterations=kwargs.get("max_iterations"),
            extra_flags=kwargs.get("extra_flags"),
        )
        info = launch_train(argv)
    except Exception as exc:
        logger.exception("train_launch failed")
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "success": True,
        "run_id": info["run_id"],
        "pid": info["pid"],
        "log_path": info["log_path"],
        "run_dir": info["run_dir"],
        "argv": info["argv"],
    }


EncreTrainLaunchTool = build_tool(
    name="train_launch",
    description=(
        "Launch a PiscesLx training run as a detached background subprocess "
        "via 'manage.py train' and return a run handle (run_id, pid, log_path) "
        "for later monitoring with train_status / train_checkpoints. Use mode="
        "'enta' to run the EnTA distill+RL hybrid loop. The process runs "
        "out-of-band so the agent is never blocked. Heavy, long-running "
        "training should always go through this tool rather than bash."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": list(_VALID_MODES),
                "description": "Training mode (default 'standard'). Use 'enta' for the distill+RL hybrid loop.",
            },
            "train_config": {"type": "string"},
            "model_size": {"type": "string"},
            "dataset": {"type": "string"},
            "resume_ckpt": {"type": "string"},
            "enta_config": {"type": "string", "description": "Path to the EnTA teachers/config yaml (default configs/teachers.yaml)."},
            "distill": {"type": "boolean", "description": "Enable distillation phase."},
            "rlhf": {"type": "boolean", "description": "Enable RLHF phase."},
            "phases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered subset of phases to run (e.g. ['data_generation','distill','rl','eval']).",
            },
            "max_iterations": {"type": "integer"},
            "extra_flags": {
                "type": "object",
                "description": "Free-form extra manage.py train flags; bool values become flag-only switches.",
            },
        },
        "required": [],
    },
    execute=_train_launch_execute,
    intents=["general", "training"],
    category="training",
    is_concurrency_safe=lambda _: True,
)
