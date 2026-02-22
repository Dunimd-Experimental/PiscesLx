#!/usr/bin/env/python3
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

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from .attacher import POPSSRunAttacher
from .controller import POPSSRunController
from .id_factory import POPSSRunIdFactory
from .store import POPSSRunStore


class POPSSRunCLI:
    def __init__(self):
        self._logger = None

    def run(self, base_args: Any, argv: List[str]) -> int:
        parser = argparse.ArgumentParser(prog="manage.py action")
        sub = parser.add_subparsers(dest="action", required=True)

        submit = sub.add_parser("submit")
        submit.add_argument("kind", choices=["train", "dataset"], help="Run kind")
        submit.add_argument("--daemon", action="store_true")
        submit.add_argument("--run_id", default=None)
        submit.add_argument("--run_name", default=None)
        submit.add_argument("--run_dir", default=None)
        submit.add_argument("--input", default=None)
        submit.add_argument("--output", default=None)
        submit.add_argument("--format", default="auto")
        submit.add_argument("--sleep", type=float, default=0.0)

        serve = sub.add_parser("serve")
        serve.add_argument("kind", choices=["infer"], help="Service kind")
        serve.add_argument("--daemon", action="store_true")
        serve.add_argument("--host", default="127.0.0.1")
        serve.add_argument("--port", type=int, default=8000)
        serve.add_argument("--run_id", default=None)
        serve.add_argument("--run_name", default=None)
        serve.add_argument("--run_dir", default=None)

        status = sub.add_parser("status")
        status.add_argument("run_id")
        status.add_argument("--run_dir", default=None)

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
        if action == "attach":
            return self._handle_attach(ns)
        if action == "logs":
            return self._handle_logs(ns)
        if action == "control":
            return self._handle_control(ns)
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
        try:
            print(json.dumps({"run_id": run_id, "run_dir": store.run_dir, "kind": ns.kind}, ensure_ascii=False))
        except Exception:
            pass

        if ns.kind == "train":
            cmd = [sys.executable, "manage.py", "train", "--run_id", run_id]
            rd = ns.run_dir or getattr(base_args, "run_dir", None)
            if rd:
                cmd.extend(["--run_dir", str(rd)])
            cmd.extend(self._build_train_forward_args(base_args))
            cmd.extend(extra)
            return self._spawn_and_maybe_attach(controller, cmd, daemon=bool(ns.daemon or getattr(base_args, "daemon", False)))

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
            return self._spawn_and_maybe_attach(controller, cmd, daemon=bool(ns.daemon or getattr(base_args, "daemon", False)))

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
        try:
            print(json.dumps({"run_id": run_id, "run_dir": store.run_dir, "kind": f"serve_{ns.kind}", "host": ns.host, "port": int(ns.port)}, ensure_ascii=False))
        except Exception:
            pass

        if ns.kind == "infer":
            cmd = [sys.executable, "manage.py", "infer", "--run_id", run_id, "--serve"]
            rd = ns.run_dir or getattr(base_args, "run_dir", None)
            if rd:
                cmd.extend(["--run_dir", str(rd)])
            cmd.extend(self._build_infer_forward_args(base_args))
            cmd.extend(["--host", str(ns.host), "--port", str(int(ns.port))])
            cmd.extend(extra)
            return self._spawn_and_maybe_attach(controller, cmd, daemon=bool(ns.daemon or getattr(base_args, "daemon", False)))

        controller.append_event("serve_unsupported", level="error", payload={"kind": ns.kind})
        controller.update_state({"status": "failed", "phase": "invalid_kind"})
        return 2

    def _spawn_and_maybe_attach(self, controller: POPSSRunController, cmd: List[str], daemon: bool) -> int:
        store = controller.store
        store.ensure()
        stdout_path = store.path("stdout")

        if daemon:
            controller.spawn(cmd, daemon=True)
            print(json.dumps({"run_id": store.run_id, "status": "running", "daemon": True, "stdout": stdout_path}, ensure_ascii=False))
            return 0

        controller.append_event("foreground_spawn", payload={"cmd": cmd})
        controller.update_state({"status": "running", "phase": "foreground"})

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            text=True,
            bufsize=1,
        )
        controller.update_state({"pid": int(p.pid)})
        controller.append_event("process_spawned", payload={"pid": int(p.pid), "daemon": False})

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
        if rc == 0:
            controller.update_state({"status": "completed", "phase": "finished", "exit_code": rc})
            controller.append_event("process_exit", payload={"exit_code": rc})
        else:
            controller.update_state({"status": "failed", "phase": "finished", "exit_code": rc})
            controller.append_event("process_exit", level="error", payload={"exit_code": rc})
        return rc

    def _handle_status(self, ns: Any) -> int:
        store = POPSSRunStore(ns.run_id, run_dir=ns.run_dir)
        state = store.read_state()
        if not state:
            print(json.dumps({"success": False, "message": "run_not_found", "run_id": ns.run_id}, ensure_ascii=False))
            return 2
        print(json.dumps({"success": True, "state": state}, ensure_ascii=False, indent=2))
        return 0

    def _handle_attach(self, ns: Any) -> int:
        store = POPSSRunStore(ns.run_id, run_dir=ns.run_dir)
        att = POPSSRunAttacher(store)
        return int(att.attach(poll_interval_s=float(ns.poll), follow=True))

    def _handle_logs(self, ns: Any) -> int:
        store = POPSSRunStore(ns.run_id, run_dir=ns.run_dir)
        p = store.path("stdout")
        if not os.path.exists(p):
            print("")
            return 0
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = int(ns.tail)
        print("".join(lines[-tail:]))
        return 0

    def _handle_control(self, ns: Any) -> int:
        store = POPSSRunStore(ns.run_id, run_dir=ns.run_dir)
        controller = POPSSRunController(store)
        action = str(getattr(ns, "control_action", "")).strip()
        if action == "kill":
            state = store.read_state()
            pid = int((state or {}).get("pid") or 0)
            ok = controller.kill_pid(pid, force=True)
            controller.append_event("control_kill", payload={"pid": pid, "ok": bool(ok)})
            if ok:
                controller.update_state({"status": "cancelled", "phase": "killed"})
            print(json.dumps({"success": bool(ok), "pid": pid}, ensure_ascii=False))
            return 0 if ok else 2

        if action in ("stop", "reload"):
            controller.enqueue_control(action, payload={})
            print(json.dumps({"success": True, "run_id": ns.run_id, "action": action}, ensure_ascii=False))
            return 0

        controller.enqueue_control(action, payload={})
        print(json.dumps({"success": True, "run_id": ns.run_id, "action": action}, ensure_ascii=False))
        return 0

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

