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

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .attacher import POPSSRunAttacher
from .controller import POPSSRunController
from .id_factory import POPSSRunIdFactory
from .store import POPSSRunStore


def _format_time(ts: str) -> str:
    if not ts:
        return "-"
    try:
        if "T" in ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.strftime("%m-%d %H:%M")
        return ts[:16] if len(ts) >= 16 else ts
    except Exception:
        return ts[:16] if len(ts) >= 16 else ts


def _status_icon(status: str) -> str:
    icons = {
        "running": "*",
        "pending": "o",
        "paused": "~",
        "completed": "+",
        "failed": "x",
        "cancelled": "-",
        "dead": "!",
    }
    return icons.get(status, ".")


def _status_color(status: str) -> str:
    colors = {
        "running": "\033[92m",
        "pending": "\033[93m",
        "paused": "\033[94m",
        "completed": "\033[90m",
        "failed": "\033[91m",
        "cancelled": "\033[90m",
        "dead": "\033[91m",
    }
    return colors.get(status, "")


def _reset_color() -> str:
    return "\033[0m"


def _print_success(msg: str):
    print(f"\033[92m+\033[0m {msg}")


def _print_error(msg: str):
    print(f"\033[91mx\033[0m {msg}")


def _print_warning(msg: str):
    print(f"\033[93m!\033[0m {msg}")


def _print_info(msg: str):
    print(f"  {msg}")


def _print_hint(msg: str):
    print(f"\033[90mHint: {msg}\033[0m")


class POPSSRunCLI:
    def __init__(self):
        self._logger = None

    def run(self, base_args: Any, argv: List[str]) -> int:
        parser = argparse.ArgumentParser(prog="manage.py action")
        sub = parser.add_subparsers(dest="action", required=True)

        submit = sub.add_parser("submit")
        submit.add_argument("kind", choices=["train", "dataset"], help="Run kind")
        submit.add_argument("--daemon", action="store_true")
        submit.add_argument("--foreground", action="store_true", help="Run in foreground mode (non-daemon)")
        submit.add_argument("--run_id", default=None)
        submit.add_argument("--run_name", default=None)
        submit.add_argument("--run_dir", default=None)
        submit.add_argument("--input", default=None)
        submit.add_argument("--output", default=None)
        submit.add_argument("--format", default="auto")
        submit.add_argument("--sleep", type=float, default=0.0)
        submit.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to allocate")
        submit.add_argument("--gpu_memory", type=int, default=0, help="Minimum GPU memory in MB (0=auto)")
        submit.add_argument("--priority", choices=["high", "normal", "low"], default="normal", help="Task priority")

        serve = sub.add_parser("serve")
        serve.add_argument("kind", choices=["infer"], help="Service kind")
        serve.add_argument("--daemon", action="store_true")
        serve.add_argument("--foreground", action="store_true", help="Run in foreground mode (non-daemon)")
        serve.add_argument("--host", default="127.0.0.1")
        serve.add_argument("--port", type=int, default=8000)
        serve.add_argument("--run_id", default=None)
        serve.add_argument("--run_name", default=None)
        serve.add_argument("--run_dir", default=None)
        serve.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to allocate")

        status = sub.add_parser("status")
        status.add_argument("run_id")
        status.add_argument("--run_dir", default=None)

        list_parser = sub.add_parser("list")
        list_parser.add_argument("--all", action="store_true", help="Show all tasks including completed")
        list_parser.add_argument("--running", action="store_true", help="Show only running tasks")
        list_parser.add_argument("--run_dir", default=None)

        attach = sub.add_parser("attach")
        attach.add_argument("run_id")
        attach.add_argument("--run_dir", default=None)
        attach.add_argument("--poll", type=float, default=0.5)

        logs = sub.add_parser("logs")
        logs.add_argument("run_id")
        logs.add_argument("--run_dir", default=None)
        logs.add_argument("--tail", type=int, default=200)

        control = sub.add_parser("control")
        control.add_argument("run_id")
        control.add_argument("control_action", choices=["pause", "resume", "cancel", "save_ckpt_now", "kill", "stop", "reload"])
        control.add_argument("--run_dir", default=None)

        worker = sub.add_parser("worker")
        worker.add_argument("kind", choices=["dataset"], help="Worker kind")
        worker.add_argument("--run_id", required=True)
        worker.add_argument("--run_dir", default=None)
        worker.add_argument("--run_name", default=None)
        worker.add_argument("--control_interval", type=float, default=0.5)
        worker.add_argument("--input", required=True)
        worker.add_argument("--output", default=None)
        worker.add_argument("--format", default="auto")
        worker.add_argument("--sleep", type=float, default=0.0)

        gpu = sub.add_parser("gpu")
        gpu.add_argument("gpu_action", choices=["list", "status", "release"], help="GPU action")
        gpu.add_argument("--gpu_id", type=int, default=None)
        gpu.add_argument("--task_id", default=None)

        queue = sub.add_parser("queue")
        queue.add_argument("queue_action", choices=["list", "clear", "stats"], help="Queue action")
        queue.add_argument("--priority", choices=["high", "normal", "low"], default=None)

        resources = sub.add_parser("resources")
        resources.add_argument("resources_action", choices=["status", "utilization"], help="Resources action")

        recover = sub.add_parser("recover")
        recover.add_argument("run_id")
        recover.add_argument("--checkpoint", default=None)
        recover.add_argument("--max_restarts", type=int, default=3)

        ns, extra = parser.parse_known_args(argv)
        action = ns.action

        if action == "submit":
            return self._handle_submit(base_args, ns, extra)
        if action == "worker":
            return self._handle_worker_dataset(ns)
        if action == "serve":
            return self._handle_serve(base_args, ns, extra)
        if action == "status":
            return self._handle_status(ns)
        if action == "list":
            return self._handle_list(ns)
        if action == "attach":
            return self._handle_attach(ns)
        if action == "logs":
            return self._handle_logs(ns)
        if action == "control":
            return self._handle_control(ns)
        if action == "gpu":
            return self._handle_gpu(ns)
        if action == "queue":
            return self._handle_queue(ns)
        if action == "resources":
            return self._handle_resources(ns)
        if action == "recover":
            return self._handle_recover(ns)
        return 1

    def _select_run_id(self, base_args: Any, override: Optional[str], prefix: str) -> str:
        v = override or getattr(base_args, "run_id", None)
        v = str(v).strip() if v is not None else ""
        if v:
            return v
        return POPSSRunIdFactory(prefix=prefix).new_id()

    def _build_train_forward_args(self, base_args: Any) -> List[str]:
        out: List[str] = []
        mapping = [
            ("model_size", "--model_size"),
            ("dataset", "--dataset"),
            ("resume_ckpt", "--resume_ckpt"),
            ("train_mode", "--train_mode"),
            ("train_config", "--train_config"),
            ("dry_run", "--dry_run"),
            ("seq_len", "--seq_len"),
            ("quant", "--quant"),
            ("no_quant", "--no_quant"),
            ("rlhf", "--rlhf"),
            ("rlhf_dataset", "--rlhf_dataset"),
            ("rlhf_lr", "--rlhf_lr"),
            ("rlhf_batch_size", "--rlhf_batch_size"),
            ("rlhf_mini_batch_size", "--rlhf_mini_batch_size"),
            ("rlhf_accum_steps", "--rlhf_accum_steps"),
            ("rlhf_epochs", "--rlhf_epochs"),
            ("rlhf_max_samples", "--rlhf_max_samples"),
            ("rlhf_max_length", "--rlhf_max_length"),
            ("model_path", "--model_path"),
        ]
        for attr, flag in mapping:
            if not hasattr(base_args, attr):
                continue
            val = getattr(base_args, attr)
            if val is None:
                continue
            if isinstance(val, bool):
                if val:
                    out.append(flag)
                continue
            s = str(val).strip()
            if not s:
                continue
            out.extend([flag, s])
        ctrl_interval = getattr(base_args, "control_interval", None)
        if ctrl_interval is not None:
            try:
                out.extend(["--control_interval", str(float(ctrl_interval))])
            except Exception:
                pass
        return out

    def _build_infer_forward_args(self, base_args: Any) -> List[str]:
        out: List[str] = []
        mapping = [
            ("model_size", "--model_size"),
            ("ckpt", "--ckpt"),
            ("infer_mode", "--infer_mode"),
            ("infer_config", "--infer_config"),
            ("model_path", "--model_path"),
            ("max_concurrency", "--max_concurrency"),
            ("request_timeout", "--request_timeout"),
        ]
        for attr, flag in mapping:
            if not hasattr(base_args, attr):
                continue
            val = getattr(base_args, attr)
            if val is None:
                continue
            if isinstance(val, bool):
                if val:
                    out.append(flag)
                continue
            s = str(val).strip()
            if not s:
                continue
            out.extend([flag, s])
        return out

    def _handle_submit(self, base_args: Any, ns: Any, extra: List[str]) -> int:
        run_id = self._select_run_id(base_args, ns.run_id, prefix=ns.kind)
        store = POPSSRunStore(run_id, run_dir=(ns.run_dir or getattr(base_args, "run_dir", None)))
        controller = POPSSRunController(store)
        spec = {
            "run_id": run_id,
            "run_name": (ns.run_name or getattr(base_args, "run_name", None) or ""),
            "type": ns.kind,
            "entry": "manage.py action submit",
            "args": {"base": vars(base_args) if hasattr(base_args, "__dict__") else {}, "extra": list(extra)},
        }
        existing_state = store.read_state() or {}
        if existing_state:
            existing_spec = store.read_spec() or {}
            merged_spec = dict(existing_spec)
            merged_spec.update(spec)
            store.write_spec(merged_spec)
            controller.append_event(
                "submit_reused_run",
                payload={"prev_phase": existing_state.get("phase"), "prev_status": existing_state.get("status")},
            )
            controller.update_state({"status": "pending", "phase": "queued"})
        else:
            controller.init_run(spec, state={"status": "pending", "phase": "queued"})

        if ns.kind == "train":
            cmd = [sys.executable, "manage.py", "train", "--run_id", run_id]
            rd = ns.run_dir or getattr(base_args, "run_dir", None)
            if rd:
                cmd.extend(["--run_dir", str(rd)])
            cmd.extend(self._build_train_forward_args(base_args))
            cmd.extend(extra)
            use_daemon = not ns.foreground and (ns.daemon or getattr(base_args, "daemon", True))
            gpu_count = getattr(ns, "gpu_count", 1) or 1
            gpu_memory_mb = getattr(ns, "gpu_memory", 0) or 0
            priority = getattr(ns, "priority", "normal") or "normal"
            return self._spawn_and_maybe_attach(
                controller, cmd, daemon=use_daemon, kind="train",
                gpu_count=gpu_count, gpu_memory_mb=gpu_memory_mb, priority=priority
            )

        if ns.kind == "dataset":
            cmd = [
                sys.executable,
                "manage.py",
                "action",
                "worker",
                "dataset",
                "--run_id",
                run_id,
                "--control_interval",
                str(float(getattr(base_args, "control_interval", 0.5) or 0.5)),
            ]
            rd = ns.run_dir or getattr(base_args, "run_dir", None)
            if rd:
                cmd.extend(["--run_dir", str(rd)])
            rn = ns.run_name or getattr(base_args, "run_name", None)
            if rn:
                cmd.extend(["--run_name", str(rn)])
            if ns.input:
                cmd.extend(["--input", str(ns.input)])
            if ns.output:
                cmd.extend(["--output", str(ns.output)])
            cmd.extend(["--format", str(ns.format or "auto")])
            try:
                sleep_s = float(ns.sleep or 0.0)
            except Exception:
                sleep_s = 0.0
            if sleep_s > 0:
                cmd.extend(["--sleep", str(sleep_s)])
            cmd.extend(extra)
            return self._spawn_and_maybe_attach(controller, cmd, daemon=bool(ns.daemon or getattr(base_args, "daemon", False)), kind="dataset")

        controller.append_event("submit_unsupported", level="error", payload={"kind": ns.kind})
        controller.update_state({"status": "failed", "phase": "invalid_kind"})
        return 2

    def _handle_serve(self, base_args: Any, ns: Any, extra: List[str]) -> int:
        run_id = self._select_run_id(base_args, ns.run_id, prefix=f"serve_{ns.kind}")
        store = POPSSRunStore(run_id, run_dir=(ns.run_dir or getattr(base_args, "run_dir", None)))
        controller = POPSSRunController(store)
        spec = {
            "run_id": run_id,
            "run_name": (ns.run_name or getattr(base_args, "run_name", None) or ""),
            "type": f"serve_{ns.kind}",
            "entry": "manage.py action serve",
            "args": {"base": vars(base_args) if hasattr(base_args, "__dict__") else {}, "extra": list(extra)},
            "service": {"host": ns.host, "port": int(ns.port)},
        }
        existing_state = store.read_state() or {}
        if existing_state:
            existing_spec = store.read_spec() or {}
            merged_spec = dict(existing_spec)
            merged_spec.update(spec)
            store.write_spec(merged_spec)
            controller.append_event(
                "serve_reused_run",
                payload={"prev_phase": existing_state.get("phase"), "prev_status": existing_state.get("status")},
            )
            controller.update_state({"status": "pending", "phase": "queued"})
        else:
            controller.init_run(spec, state={"status": "pending", "phase": "queued"})

        if ns.kind == "infer":
            cmd = [sys.executable, "manage.py", "infer", "--run_id", run_id, "--serve"]
            rd = ns.run_dir or getattr(base_args, "run_dir", None)
            if rd:
                cmd.extend(["--run_dir", str(rd)])
            cmd.extend(self._build_infer_forward_args(base_args))
            cmd.extend(["--host", str(ns.host), "--port", str(int(ns.port))])
            cmd.extend(extra)
            use_daemon = not ns.foreground and (ns.daemon or getattr(base_args, "daemon", True))
            gpu_count = getattr(ns, "gpu_count", 1) or 1
            return self._spawn_and_maybe_attach(
                controller, cmd, daemon=use_daemon, kind="serve",
                host=ns.host, port=int(ns.port), gpu_count=gpu_count
            )

        controller.append_event("serve_unsupported", level="error", payload={"kind": ns.kind})
        controller.update_state({"status": "failed", "phase": "invalid_kind"})
        return 2

    def _spawn_and_maybe_attach(self, controller: POPSSRunController, cmd: List[str], daemon: bool, kind: str = "task", host: str = None, port: int = None, gpu_count: int = 1, gpu_memory_mb: int = 0, priority: str = "normal") -> int:
        store = controller.store
        store.ensure()
        stdout_path = store.path("stdout")
        
        allocated_gpus = []
        try:
            allocated_gpus = controller.allocate_gpus(
                gpu_count=gpu_count,
                min_memory_mb=gpu_memory_mb,
                priority=priority,
            )
        except Exception as e:
            print()
            _print_error(f"Failed to allocate GPUs: {e}")
            print()
            return 2
        
        cuda_visible_devices = ",".join(str(g) for g in allocated_gpus) if allocated_gpus else ""
        
        env = os.environ.copy()
        if cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
        controller.update_state({
            "allocated_gpus": allocated_gpus,
            "cuda_visible_devices": cuda_visible_devices,
        })

        if daemon:
            pid = controller.spawn(cmd, daemon=True, env=env)
            print()
            _print_success("Task submitted (background)")
            print()
            _print_info(f"Run ID: {store.run_id}")
            _print_info(f"PID:    {pid}")
            if allocated_gpus:
                _print_info(f"GPUs:   {cuda_visible_devices}")
            if host and port:
                _print_info(f"URL:    http://{host}:{port}")
            _print_info(f"Log:    {stdout_path}")
            print()
            _print_hint(f"python manage.py action status {store.run_id}")
            _print_hint(f"python manage.py action logs {store.run_id}")
            _print_hint(f"python manage.py action control {store.run_id} stop")
            print()
            return 0

        controller.append_event("foreground_spawn", payload={"cmd": cmd, "gpus": allocated_gpus})
        controller.update_state({"status": "running", "phase": "foreground"})

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            text=True,
            bufsize=1,
            env=env,
        )
        controller.update_state({"pid": int(p.pid)})
        controller.append_event("process_spawned", payload={"pid": int(p.pid), "daemon": False, "gpus": allocated_gpus})

        print()
        _print_success("Task started (foreground)")
        print()
        _print_info(f"Run ID: {store.run_id}")
        _print_info(f"PID:    {p.pid}")
        if allocated_gpus:
            _print_info(f"GPUs:   {cuda_visible_devices}")
        print()
        _print_warning("Press Ctrl+C to stop")
        print()

        with open(stdout_path, "a", encoding="utf-8") as log:
            while True:
                line = p.stdout.readline() if p.stdout is not None else ""
                if not line:
                    if p.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue
                print(line, end="")
                try:
                    log.write(line)
                    log.flush()
                except Exception:
                    pass

        rc = int(p.wait())
        controller.release_gpus()
        print()
        if rc == 0:
            _print_success("Task completed")
            controller.update_state({"status": "completed", "phase": "finished", "exit_code": rc})
            controller.append_event("process_exit", payload={"exit_code": rc})
        else:
            _print_error(f"Task failed (exit code: {rc})")
            controller.update_state({"status": "failed", "phase": "finished", "exit_code": rc})
            controller.append_event("process_exit", level="error", payload={"exit_code": rc})
        print()
        return rc

    def _handle_status(self, ns: Any) -> int:
        store = POPSSRunStore(ns.run_id, run_dir=ns.run_dir)
        state = store.read_state()
        if not state:
            print()
            _print_error(f"Task not found: {ns.run_id}")
            print()
            return 2
        
        pid = state.get("pid")
        status = state.get("status", "")
        if pid and status == "running":
            try:
                os.kill(int(pid), 0)
            except Exception:
                state["status"] = "dead"
                state["phase"] = "process_lost"
                controller = POPSSRunController(store)
                controller.update_state({"status": "dead", "phase": "process_lost"})
                controller.append_event("process_lost", level="warning", payload={"pid": pid})
        
        status_val = state.get("status", "-")
        icon = _status_icon(status_val)
        color = _status_color(status_val)
        reset = _reset_color()
        
        print()
        print(f"Task: {ns.run_id}")
        print("+- " + f"Status: {color}{icon} {status_val}{reset}")
        print("+- " + f"Phase:  {state.get('phase', '-')}")
        print("+- " + f"PID:    {state.get('pid', '-')}")
        print("+- " + f"Created: {_format_time(state.get('created_at', ''))}")
        print("`- " + f"Updated: {_format_time(state.get('updated_at', ''))}")
        print()
        return 0

    def _handle_list(self, ns: Any) -> int:
        run_dir = ns.run_dir or ".pisceslx/runs"
        if not os.path.exists(run_dir):
            print()
            _print_info("No tasks found")
            print()
            return 0
        
        tasks = []
        for name in os.listdir(run_dir):
            task_dir = os.path.join(run_dir, name)
            if not os.path.isdir(task_dir):
                continue
            state_file = os.path.join(task_dir, "state.json")
            if not os.path.exists(state_file):
                continue
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                status_val = state.get("status", "unknown")
                if ns.running and status_val != "running":
                    continue
                if not ns.all and status_val in ("completed", "failed", "cancelled"):
                    continue
                tasks.append({
                    "run_id": state.get("run_id", name),
                    "status": status_val,
                    "phase": state.get("phase", "-"),
                    "pid": state.get("pid", "-"),
                    "created_at": state.get("created_at", ""),
                })
            except Exception:
                continue
        
        if not tasks:
            print()
            if ns.running:
                _print_info("No running tasks")
            elif ns.all:
                _print_info("No tasks found")
            else:
                _print_info("No active tasks (use --all to see all)")
            print()
            return 0
        
        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        print()
        print(f"Task List ({len(tasks)} total)")
        print()
        print(f"{'ID':<36} {'Status':<10} {'Phase':<12} {'PID':<8} {'Created'}")
        print("-" * 80)
        
        for task in tasks:
            run_id = task["run_id"]
            status_val = task["status"]
            phase = task["phase"] or "-"
            pid = str(task["pid"]) if task["pid"] else "-"
            created = _format_time(task["created_at"])
            
            icon = _status_icon(status_val)
            color = _status_color(status_val)
            reset = _reset_color()
            
            status_display = f"{color}{icon} {status_val}{reset}"
            
            if len(run_id) > 34:
                run_id_display = run_id[:34] + ".."
            else:
                run_id_display = run_id
            
            print(f"{run_id_display:<36} {status_display:<19} {phase:<12} {pid:<8} {created}")
        
        print()
        return 0

    def _handle_attach(self, ns: Any) -> int:
        store = POPSSRunStore(ns.run_id, run_dir=ns.run_dir)
        att = POPSSRunAttacher(store)
        return int(att.attach(poll_interval_s=float(ns.poll), follow=True))

    def _handle_logs(self, ns: Any) -> int:
        store = POPSSRunStore(ns.run_id, run_dir=ns.run_dir)
        p = store.path("stdout")
        if not os.path.exists(p):
            print()
            _print_info("No logs available")
            print()
            return 0
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = int(ns.tail)
        
        print()
        print("=" * 70)
        print(f"Task Log: {ns.run_id} (last {tail} lines)")
        print("=" * 70)
        print("".join(lines[-tail:]))
        print("=" * 70)
        print()
        return 0

    def _handle_control(self, ns: Any) -> int:
        store = POPSSRunStore(ns.run_id, run_dir=ns.run_dir)
        controller = POPSSRunController(store)
        action = str(getattr(ns, "control_action", "")).strip()
        state = store.read_state()
        pid = int((state or {}).get("pid") or 0)
        
        if action == "kill":
            print()
            _print_warning(f"Force killing process {pid} (task: {ns.run_id})")
            ok = controller.kill_pid(pid, force=True)
            controller.append_event("control_kill", payload={"pid": pid, "ok": bool(ok)})
            if ok:
                controller.update_state({"status": "cancelled", "phase": "killed"})
                _print_success("Process terminated")
                _print_warning("Unsaved progress will be lost")
            else:
                _print_error("Failed to terminate process")
            print()
            return 0 if ok else 2

        if action == "stop":
            controller.enqueue_control(action, payload={})
            print()
            _print_success(f"Sent stop signal to task {ns.run_id}")
            if pid:
                _print_info(f"PID: {pid}")
            _print_info("Task will save checkpoint and exit gracefully")
            print()
            return 0

        if action == "pause":
            controller.enqueue_control(action, payload={})
            print()
            _print_success(f"Sent pause signal to task {ns.run_id}")
            _print_hint("Use resume to continue the task")
            print()
            return 0

        if action == "resume":
            controller.enqueue_control(action, payload={})
            print()
            _print_success(f"Sent resume signal to task {ns.run_id}")
            print()
            return 0

        if action == "save_ckpt_now":
            controller.enqueue_control(action, payload={})
            print()
            _print_success(f"Sent save_ckpt_now signal to task {ns.run_id}")
            print()
            return 0

        controller.enqueue_control(action, payload={})
        print()
        _print_success(f"Sent {action} signal to task {ns.run_id}")
        print()
        return 0

    def _handle_gpu(self, ns: Any) -> int:
        from .gpu_scheduler import POPSSGPUScheduler
        
        scheduler = POPSSGPUScheduler()
        action = ns.gpu_action
        
        if action == "list":
            gpu_info = scheduler.get_gpu_info()
            util = scheduler.get_utilization()
            
            print()
            print(f"GPU Status ({util['total_gpus']} total, {util['available_gpus']} available)")
            print()
            
            if not gpu_info:
                _print_info("No GPUs detected")
                print()
                return 0
            
            print(f"{'ID':<4} {'Name':<30} {'Memory':<15} {'Util':<8} {'Status'}")
            print("-" * 75)
            
            for gpu_id, info in sorted(gpu_info.items()):
                mem = f"{info.free_memory_mb}/{info.total_memory_mb}MB"
                util_pct = f"{info.utilization_percent}%"
                status = "free" if info.is_available else f"used by {info.allocated_to}"
                
                print(f"{gpu_id:<4} {info.name[:28]:<30} {mem:<15} {util_pct:<8} {status}")
            
            print()
            return 0
        
        if action == "status":
            gpu_id = ns.gpu_id
            if gpu_id is not None:
                gpu_info = scheduler.get_gpu_info(gpu_id)
                if not gpu_info:
                    print()
                    _print_error(f"GPU {gpu_id} not found")
                    print()
                    return 2
                
                info = gpu_info[gpu_id]
                print()
                print(f"GPU {gpu_id}: {info.name}")
                print("+- " + f"Memory: {info.free_memory_mb}/{info.total_memory_mb}MB free")
                print("+- " + f"Utilization: {info.utilization_percent}%")
                print("+- " + f"Temperature: {info.temperature_c}C")
                print("+- " + f"Power: {info.power_usage_w}/{info.power_limit_w}W")
                print("`- " + f"Status: {'available' if info.is_available else 'allocated to ' + str(info.allocated_to)}")
                print()
                return 0
            
            util = scheduler.get_utilization()
            print()
            print("GPU Utilization Summary")
            print("+- " + f"Total GPUs: {util['total_gpus']}")
            print("+- " + f"Available: {util['available_gpus']}")
            print("+- " + f"Allocated: {util['allocated_gpus']}")
            print("+- " + f"Memory: {util['used_memory_mb']}/{util['total_memory_mb']}MB ({util['memory_utilization_percent']:.1f}%)")
            print("`- " + f"Active tasks: {util['allocations']}")
            print()
            return 0
        
        if action == "release":
            task_id = ns.task_id
            if not task_id:
                print()
                _print_error("Task ID required for release")
                print()
                return 2
            
            released = scheduler.release(task_id)
            if released:
                print()
                _print_success(f"Released GPUs {released} from task {task_id}")
                print()
                return 0
            else:
                print()
                _print_error(f"No GPUs allocated to task {task_id}")
                print()
                return 2
        
        return 1

    def _handle_queue(self, ns: Any) -> int:
        from .queue import POPSSTaskQueue
        
        queue = POPSSTaskQueue()
        action = ns.queue_action
        
        if action == "list":
            tasks = queue.get_all_tasks()
            stats = queue.get_stats()
            
            print()
            print(f"Task Queue ({stats['total_queued']} pending, {stats['running']} running)")
            print()
            
            if not tasks:
                _print_info("Queue is empty")
                print()
                return 0
            
            print(f"{'ID':<36} {'Type':<10} {'Priority':<10} {'Created'}")
            print("-" * 75)
            
            for task in tasks:
                created = _format_time(task.created_at)
                print(f"{task.task_id:<36} {task.task_type:<10} {task.priority:<10} {created}")
            
            print()
            return 0
        
        if action == "clear":
            priority = ns.priority
            count = queue.clear(priority)
            print()
            _print_success(f"Cleared {count} tasks from queue")
            print()
            return 0
        
        if action == "stats":
            stats = queue.get_stats()
            print()
            print("Queue Statistics")
            print("+- " + f"High priority: {stats['high_priority']}")
            print("+- " + f"Normal priority: {stats['normal_priority']}")
            print("+- " + f"Low priority: {stats['low_priority']}")
            print("+- " + f"Running: {stats['running']}")
            print("`- " + f"Completed: {stats['completed']}")
            print()
            return 0
        
        return 1

    def _handle_resources(self, ns: Any) -> int:
        from .resources import POPSSResourceLimits
        
        limits = POPSSResourceLimits()
        action = ns.resources_action
        
        if action == "status":
            system_info = limits.get_system_info()
            
            print()
            print("System Resources")
            print("+- " + f"CPU Cores: {system_info['cpu_count']}")
            print("+- " + f"CPU Usage: {system_info['cpu_percent']:.1f}%")
            print("+- " + f"Memory: {system_info['memory_available_mb']}/{system_info['memory_total_mb']}MB available")
            print("+- " + f"Memory Usage: {system_info['memory_percent']:.1f}%")
            print("+- " + f"Storage: {system_info['storage_available_mb']}/{system_info['storage_total_mb']}MB available")
            print("`- " + f"Storage Usage: {system_info['storage_percent']:.1f}%")
            print()
            return 0
        
        if action == "utilization":
            util = limits.get_utilization()
            system = util['system']
            allocated = util['allocated']
            available = util['available']
            
            print()
            print("Resource Utilization")
            print()
            print("System:")
            print("  " + f"CPU: {system['cpu_count']} cores ({system['cpu_percent']:.1f}% used)")
            print("  " + f"Memory: {system['memory_total_mb']}MB total ({system['memory_percent']:.1f}% used)")
            print("  " + f"Storage: {system['storage_total_mb']}MB total ({system['storage_percent']:.1f}% used)")
            print()
            print("Allocated:")
            print("  " + f"Tasks: {allocated['tasks']}")
            print("  " + f"CPU Cores: {allocated['cpu_cores']}")
            print("  " + f"Memory: {allocated['memory_mb']}MB")
            print()
            print("Available:")
            print("  " + f"CPU Cores: {available['cpu_cores']}")
            print("  " + f"Memory: {available['memory_mb']}MB")
            print()
            return 0
        
        return 1

    def _handle_recover(self, ns: Any) -> int:
        from .recovery import POPSSTaskRecovery
        from .controller import POPSSRunController
        
        store = POPSSRunStore(ns.run_id, run_dir=ns.run_dir)
        controller = POPSSRunController(store)
        recovery = POPSSTaskRecovery(controller)
        
        recovery.register_task(ns.run_id, ns.run_id, max_restarts=ns.max_restarts)
        
        if not recovery.check_crashed(ns.run_id, store):
            print()
            _print_info(f"Task {ns.run_id} is not crashed")
            print()
            return 0
        
        checkpoint_path = ns.checkpoint
        if checkpoint_path:
            from .recovery import POPSSCheckpointInfo
            checkpoint = POPSSCheckpointInfo(
                checkpoint_path=checkpoint_path,
                step=0,
                epoch=0,
                loss=0.0,
                created_at="",
                size_mb=0,
            )
        else:
            checkpoint = recovery.find_latest_checkpoint(ns.run_id)
        
        if checkpoint is None:
            print()
            _print_error(f"No checkpoint found for task {ns.run_id}")
            print()
            return 2
        
        success, message = recovery.restart_task(ns.run_id, checkpoint, store)
        
        print()
        if success:
            _print_success(message)
            _print_info(f"Checkpoint: {checkpoint.checkpoint_path}")
            _print_info(f"Step: {checkpoint.step}")
        else:
            _print_error(message)
        print()
        
        return 0 if success else 2

    def _handle_worker_dataset(self, ns: Any) -> int:
        run_id = str(ns.run_id).strip()
        store = POPSSRunStore(run_id, run_dir=ns.run_dir)
        controller = POPSSRunController(store)
        controller.init_run(
            {
                "run_id": run_id,
                "run_name": str(ns.run_name or "").strip(),
                "type": "dataset",
                "input": str(ns.input),
                "output": str(ns.output or ""),
                "format": str(ns.format or "auto"),
            },
            state={"status": "running", "phase": "running", "pid": int(os.getpid())},
        )

        last_seq = int((store.read_state() or {}).get("last_control_seq") or 0)
        paused = False
        cancelled = False
        ctrl_interval = float(ns.control_interval or 0.5)
        sleep_s = 0.0
        try:
            sleep_s = float(ns.sleep or 0.0)
        except Exception:
            sleep_s = 0.0

        def poll():
            nonlocal last_seq, paused, cancelled
            last_seq, items = controller.poll_controls(last_seq)
            controller.update_state({"last_control_seq": int(last_seq)})
            for obj in items:
                act = str(obj.get("action") or "")
                if act == "pause":
                    paused = True
                    controller.update_state({"status": "paused", "phase": "paused"})
                    controller.append_event("dataset_pause_requested", payload={"seq": obj.get("seq")})
                elif act == "resume":
                    paused = False
                    controller.update_state({"status": "running", "phase": "running"})
                    controller.append_event("dataset_resume_requested", payload={"seq": obj.get("seq")})
                elif act in ("cancel", "kill"):
                    cancelled = True
                    controller.update_state({"status": "cancelled", "phase": "cancelled"})
                    controller.append_event("dataset_cancel_requested", payload={"seq": obj.get("seq"), "action": act})

        inp = str(ns.input)
        if not os.path.exists(inp):
            controller.update_state({"status": "failed", "phase": "failed", "error": "input_not_found"})
            controller.append_event("dataset_failed", level="error", payload={"error": "input_not_found", "input": inp})
            return 2

        total = os.path.getsize(inp)
        processed = 0
        ok = 0
        bad = 0
        fmt = str(ns.format or "auto").lower()
        if fmt == "auto":
            if inp.lower().endswith(".jsonl"):
                fmt = "jsonl"
            else:
                fmt = "text"

        out_path = str(ns.output or "").strip()
        out_fh = None
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            out_fh = open(out_path, "w", encoding="utf-8")

        try:
            with open(inp, "rb") as f:
                for line in f:
                    processed += len(line)
                    poll()
                    while paused and not cancelled:
                        time.sleep(ctrl_interval)
                        poll()
                    if cancelled:
                        break

                    if fmt == "jsonl":
                        try:
                            json.loads(line.decode("utf-8", errors="strict"))
                            ok += 1
                        except Exception:
                            bad += 1
                    else:
                        if line.strip():
                            ok += 1
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                    if (ok + bad) % 200 == 0:
                        progress = int(processed * 100 / max(total, 1))
                        controller.append_metric(
                            {
                                "processed_bytes": processed,
                                "total_bytes": total,
                                "progress": progress,
                                "ok": ok,
                                "bad": bad,
                            }
                        )
                        controller.update_state(
                            {
                                "status": "running" if not paused else "paused",
                                "phase": "running" if not paused else "paused",
                                "processed_bytes": processed,
                                "total_bytes": total,
                                "progress": progress,
                                "ok": ok,
                                "bad": bad,
                            }
                        )

            if cancelled:
                controller.update_state({"status": "cancelled", "phase": "cancelled", "ok": ok, "bad": bad})
                controller.append_event("dataset_cancelled", payload={"ok": ok, "bad": bad})
                return 0

            summary = {"input": inp, "format": fmt, "ok": ok, "bad": bad, "bytes": processed}
            if out_fh is not None:
                out_fh.write(json.dumps(summary, ensure_ascii=False, indent=2))
                out_fh.flush()
                controller.record_artifact("output", out_path)

            controller.append_metric(
                {
                    "processed_bytes": processed,
                    "total_bytes": total,
                    "progress": 100,
                    "ok": ok,
                    "bad": bad,
                }
            )
            controller.update_state({"status": "completed", "phase": "completed", "progress": 100, "ok": ok, "bad": bad})
            controller.append_event("dataset_completed", payload=summary)
            return 0
        finally:
            try:
                if out_fh is not None:
                    out_fh.close()
            except Exception:
                pass
