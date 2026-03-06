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
import sys
import time
import threading
import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
import psutil
import pynvml

from .store import POPSSRunStore
from .controller import POPSSRunController


class POPSSResourceSnapshot:
    def __init__(
        self,
        ts: str,
        pid: int,
        cpu_percent: float,
        memory_mb: float,
        memory_percent: float,
        gpu_memory_mb: Optional[float] = None,
        gpu_utilization: Optional[float] = None,
        gpu_memory_total_mb: Optional[float] = None,
        num_threads: Optional[int] = None,
        num_fds: Optional[int] = None,
        io_read_mb: Optional[float] = None,
        io_write_mb: Optional[float] = None,
    ):
        self.ts = ts
        self.pid = pid
        self.cpu_percent = cpu_percent
        self.memory_mb = memory_mb
        self.memory_percent = memory_percent
        self.gpu_memory_mb = gpu_memory_mb
        self.gpu_utilization = gpu_utilization
        self.gpu_memory_total_mb = gpu_memory_total_mb
        self.num_threads = num_threads
        self.num_fds = num_fds
        self.io_read_mb = io_read_mb
        self.io_write_mb = io_write_mb

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "ts": self.ts,
            "pid": self.pid,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
        }
        if self.gpu_memory_mb is not None:
            d["gpu_memory_mb"] = self.gpu_memory_mb
        if self.gpu_utilization is not None:
            d["gpu_utilization"] = self.gpu_utilization
        if self.gpu_memory_total_mb is not None:
            d["gpu_memory_total_mb"] = self.gpu_memory_total_mb
        if self.num_threads is not None:
            d["num_threads"] = self.num_threads
        if self.num_fds is not None:
            d["num_fds"] = self.num_fds
        if self.io_read_mb is not None:
            d["io_read_mb"] = self.io_read_mb
        if self.io_write_mb is not None:
            d["io_write_mb"] = self.io_write_mb
        return d


class POPSSResourceMonitor:
    def __init__(
        self,
        store: POPSSRunStore,
        interval_s: float = 5.0,
        enable_gpu: bool = True,
        enable_io: bool = True,
    ):
        self._store = store
        self._controller = POPSSRunController(store)
        self._interval_s = max(0.5, float(interval_s))
        self._enable_gpu = bool(enable_gpu)
        self._enable_io = bool(enable_io)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_snapshot: Optional[POPSSResourceSnapshot] = None
        self._history: List[POPSSResourceSnapshot] = []
        self._max_history = 1000
        self._nvml_initialized = False
        self._gpu_handle = None
        self._last_io_read = 0.0
        self._last_io_write = 0.0
        self._callbacks: List[Callable[[POPSSResourceSnapshot], None]] = []

    @property
    def last_snapshot(self) -> Optional[POPSSResourceSnapshot]:
        return self._last_snapshot

    @property
    def history(self) -> List[POPSSResourceSnapshot]:
        return list(self._history)

    def add_callback(self, callback: Callable[[POPSSResourceSnapshot], None]) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[POPSSResourceSnapshot], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _init_nvml(self) -> bool:
        if not self._enable_gpu:
            return False
        if self._nvml_initialized:
            return self._gpu_handle is not None
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count <= 0:
                return False
            gpu_id = 0
            env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if env_gpu:
                try:
                    gpu_id = int(env_gpu.split(",")[0].strip())
                except Exception:
                    gpu_id = 0
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self._nvml_initialized = True
            return True
        except Exception:
            self._nvml_initialized = False
            self._gpu_handle = None
            return False

    def _shutdown_nvml(self) -> None:
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml_initialized = False
        self._gpu_handle = None

    def _get_gpu_info(self) -> Dict[str, Optional[float]]:
        result = {
            "gpu_memory_mb": None,
            "gpu_utilization": None,
            "gpu_memory_total_mb": None,
        }
        if not self._init_nvml() or self._gpu_handle is None:
            return result
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            result["gpu_memory_mb"] = float(mem_info.used) / (1024 * 1024)
            result["gpu_memory_total_mb"] = float(mem_info.total) / (1024 * 1024)
        except Exception:
            pass
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            result["gpu_utilization"] = float(util.gpu)
        except Exception:
            pass
        return result

    def _get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        try:
            proc = psutil.Process(pid)
            with proc.oneshot():
                cpu_percent = proc.cpu_percent(interval=None)
                mem_info = proc.memory_info()
                memory_mb = float(mem_info.rss) / (1024 * 1024)
                try:
                    memory_percent = proc.memory_percent()
                except Exception:
                    memory_percent = 0.0
                num_threads = proc.num_threads()
                num_fds = None
                if platform.system() != "Windows":
                    try:
                        num_fds = proc.num_fds()
                    except Exception:
                        pass
                io_read_mb = None
                io_write_mb = None
                if self._enable_io:
                    try:
                        io_counters = proc.io_counters()
                        io_read_mb = float(io_counters.read_bytes) / (1024 * 1024)
                        io_write_mb = float(io_counters.write_bytes) / (1024 * 1024)
                    except Exception:
                        pass
                return {
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                    "memory_percent": memory_percent,
                    "num_threads": num_threads,
                    "num_fds": num_fds,
                    "io_read_mb": io_read_mb,
                    "io_write_mb": io_write_mb,
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
        except Exception:
            return None

    def snapshot(self, pid: Optional[int] = None) -> Optional[POPSSResourceSnapshot]:
        if pid is None:
            state = self._store.read_state()
            pid = int((state or {}).get("pid") or 0)
        if pid <= 0:
            return None
        proc_info = self._get_process_info(pid)
        if proc_info is None:
            return None
        gpu_info = self._get_gpu_info() if self._enable_gpu else {}
        ts = datetime.now(timezone.utc).isoformat()
        snapshot = POPSSResourceSnapshot(
            ts=ts,
            pid=pid,
            cpu_percent=proc_info.get("cpu_percent", 0.0),
            memory_mb=proc_info.get("memory_mb", 0.0),
            memory_percent=proc_info.get("memory_percent", 0.0),
            gpu_memory_mb=gpu_info.get("gpu_memory_mb"),
            gpu_utilization=gpu_info.get("gpu_utilization"),
            gpu_memory_total_mb=gpu_info.get("gpu_memory_total_mb"),
            num_threads=proc_info.get("num_threads"),
            num_fds=proc_info.get("num_fds"),
            io_read_mb=proc_info.get("io_read_mb"),
            io_write_mb=proc_info.get("io_write_mb"),
        )
        with self._lock:
            self._last_snapshot = snapshot
            self._history.append(snapshot)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
        return snapshot

    def _emit_metrics(self, snapshot: POPSSResourceSnapshot) -> None:
        metric: Dict[str, Any] = {
            "type": "resource",
            "cpu_percent": snapshot.cpu_percent,
            "memory_mb": snapshot.memory_mb,
            "memory_percent": snapshot.memory_percent,
        }
        if snapshot.gpu_memory_mb is not None:
            metric["gpu_memory_mb"] = snapshot.gpu_memory_mb
        if snapshot.gpu_utilization is not None:
            metric["gpu_utilization"] = snapshot.gpu_utilization
        if snapshot.num_threads is not None:
            metric["num_threads"] = snapshot.num_threads
        self._controller.append_metric(metric)

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                snapshot = self.snapshot()
                if snapshot is not None:
                    self._emit_metrics(snapshot)
                    for cb in self._callbacks:
                        try:
                            cb(snapshot)
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(self._interval_s)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._shutdown_nvml()

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._history:
                return {"samples": 0}
            cpus = [s.cpu_percent for s in self._history if s.cpu_percent is not None]
            mems = [s.memory_mb for s in self._history if s.memory_mb is not None]
            gpumems = [s.gpu_memory_mb for s in self._history if s.gpu_memory_mb is not None]
            summary: Dict[str, Any] = {
                "samples": len(self._history),
                "cpu_avg": sum(cpus) / len(cpus) if cpus else 0.0,
                "cpu_max": max(cpus) if cpus else 0.0,
                "memory_avg_mb": sum(mems) / len(mems) if mems else 0.0,
                "memory_max_mb": max(mems) if mems else 0.0,
            }
            if gpumems:
                summary["gpu_memory_avg_mb"] = sum(gpumems) / len(gpumems)
                summary["gpu_memory_max_mb"] = max(gpumems)
            return summary


class POPSSHeartbeatMonitor:
    def __init__(
        self,
        store: POPSSRunStore,
        interval_s: float = 10.0,
        timeout_s: float = 60.0,
        max_missed: int = 3,
    ):
        self._store = store
        self._controller = POPSSRunController(store)
        self._interval_s = max(1.0, float(interval_s))
        self._timeout_s = max(self._interval_s * 2, float(timeout_s))
        self._max_missed = max(1, int(max_missed))
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_heartbeat_ts: Optional[str] = None
        self._missed_count = 0
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

    def add_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def beat(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._controller.update_state({
            "last_heartbeat": ts,
            "heartbeat_metadata": metadata or {},
        })
        self._controller.append_event("heartbeat", payload={"ts": ts, "metadata": metadata or {}})
        self._last_heartbeat_ts = ts
        self._missed_count = 0

    def _check_heartbeat(self) -> Optional[str]:
        state = self._store.read_state()
        if not state:
            return None
        last_hb = state.get("last_heartbeat")
        if not last_hb:
            return None
        try:
            last_dt = datetime.fromisoformat(str(last_hb).replace("Z", "+00:00"))
            now_dt = datetime.now(timezone.utc)
            elapsed = (now_dt - last_dt).total_seconds()
            if elapsed > self._timeout_s:
                return "timeout"
            return None
        except Exception:
            return None

    def _emit_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        self._controller.append_event(f"heartbeat_{alert_type}", level="warning", payload=details)
        for cb in self._callbacks:
            try:
                cb(alert_type, details)
            except Exception:
                pass

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                status = self._check_heartbeat()
                if status == "timeout":
                    self._missed_count += 1
                    if self._missed_count >= self._max_missed:
                        self._emit_alert("dead", {"missed_count": self._missed_count})
                        self._controller.update_state({"status": "dead", "phase": "heartbeat_timeout"})
                    else:
                        self._emit_alert("missed", {"missed_count": self._missed_count})
                else:
                    if self._missed_count > 0:
                        self._missed_count = 0
                        self._controller.update_state({"status": "running"})
            except Exception:
                pass
            time.sleep(self._interval_s)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class POPSSRunManager:
    def __init__(self, runs_root: Optional[str] = None):
        if runs_root is None:
            from pathlib import Path
            runs_root = str(Path(".pisceslx") / "runs")
        self._runs_root = str(runs_root)
        self._ensure_root()

    def _ensure_root(self) -> None:
        os.makedirs(self._runs_root, exist_ok=True)

    def _run_dir(self, run_id: str) -> str:
        return os.path.join(self._runs_root, run_id)

    def list_runs(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        self._ensure_root()
        runs: List[Dict[str, Any]] = []
        try:
            entries = os.listdir(self._runs_root)
        except Exception:
            return runs
        for entry in entries:
            run_dir = os.path.join(self._runs_root, entry)
            if not os.path.isdir(run_dir):
                continue
            state_file = os.path.join(run_dir, "state.json")
            if not os.path.exists(state_file):
                continue
            try:
                import json
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f) or {}
            except Exception:
                continue
            if status is not None and state.get("status") != status:
                continue
            runs.append({
                "run_id": entry,
                "run_dir": run_dir,
                "status": state.get("status"),
                "phase": state.get("phase"),
                "created_at": state.get("created_at"),
                "updated_at": state.get("updated_at"),
                "pid": state.get("pid"),
            })
            if len(runs) >= limit:
                break
        runs.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return runs

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        run_dir = self._run_dir(run_id)
        if not os.path.isdir(run_dir):
            return None
        store = POPSSRunStore(run_id, run_dir=run_dir)
        state = store.read_state()
        spec = store.read_spec()
        if not state:
            return None
        return {
            "run_id": run_id,
            "run_dir": run_dir,
            "state": state,
            "spec": spec,
        }

    def delete_run(self, run_id: str, force: bool = False) -> bool:
        run_dir = self._run_dir(run_id)
        if not os.path.isdir(run_dir):
            return False
        store = POPSSRunStore(run_id, run_dir=run_dir)
        state = store.read_state()
        if not force:
            status = (state or {}).get("status")
            if status in ("running", "pending"):
                return False
        return store.delete()

    def kill_run(self, run_id: str) -> bool:
        run_dir = self._run_dir(run_id)
        if not os.path.isdir(run_dir):
            return False
        store = POPSSRunStore(run_id, run_dir=run_dir)
        controller = POPSSRunController(store)
        state = store.read_state()
        pid = int((state or {}).get("pid") or 0)
        if pid <= 0:
            return False
        return controller.kill_pid(pid, force=True)

    def get_stats(self) -> Dict[str, Any]:
        runs = self.list_runs(limit=10000)
        status_counts: Dict[str, int] = {}
        for run in runs:
            st = run.get("status") or "unknown"
            status_counts[st] = status_counts.get(st, 0) + 1
        return {
            "total_runs": len(runs),
            "status_counts": status_counts,
            "runs_root": self._runs_root,
        }


class POPSSRunCleaner:
    def __init__(
        self,
        runs_root: Optional[str] = None,
        max_age_days: int = 30,
        max_runs: int = 1000,
        keep_statuses: Optional[List[str]] = None,
    ):
        if runs_root is None:
            from pathlib import Path
            runs_root = str(Path(".pisceslx") / "runs")
        self._runs_root = str(runs_root)
        self._max_age_days = max(1, int(max_age_days))
        self._max_runs = max(10, int(max_runs))
        self._keep_statuses = list(keep_statuses) if keep_statuses else ["running", "pending"]

    def _run_dir(self, run_id: str) -> str:
        return os.path.join(self._runs_root, run_id)

    def _get_run_age_days(self, state: Dict[str, Any]) -> Optional[float]:
        created = state.get("created_at")
        if not created:
            return None
        try:
            created_dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            now_dt = datetime.now(timezone.utc)
            return (now_dt - created_dt).total_seconds() / 86400.0
        except Exception:
            return None

    def scan(self) -> Dict[str, List[Dict[str, Any]]]:
        os.makedirs(self._runs_root, exist_ok=True)
        to_clean: List[Dict[str, Any]] = []
        to_keep: List[Dict[str, Any]] = []
        try:
            entries = os.listdir(self._runs_root)
        except Exception:
            return {"to_clean": [], "to_keep": []}
        for entry in entries:
            run_dir = os.path.join(self._runs_root, entry)
            if not os.path.isdir(run_dir):
                continue
            state_file = os.path.join(run_dir, "state.json")
            if not os.path.exists(state_file):
                to_clean.append({"run_id": entry, "reason": "no_state"})
                continue
            try:
                import json
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f) or {}
            except Exception:
                to_clean.append({"run_id": entry, "reason": "corrupt_state"})
                continue
            status = state.get("status")
            if status in self._keep_statuses:
                to_keep.append({"run_id": entry, "status": status})
                continue
            age_days = self._get_run_age_days(state)
            if age_days is not None and age_days > self._max_age_days:
                to_clean.append({"run_id": entry, "reason": "age", "age_days": age_days})
                continue
            to_keep.append({"run_id": entry, "status": status, "age_days": age_days})
        to_keep.sort(key=lambda x: x.get("age_days") or 0, reverse=True)
        while len(to_keep) > self._max_runs:
            removed = to_keep.pop()
            removed["reason"] = "excess"
            to_clean.append(removed)
        return {"to_clean": to_clean, "to_keep": to_keep}

    def clean(self, dry_run: bool = False) -> Dict[str, Any]:
        scan_result = self.scan()
        cleaned: List[str] = []
        errors: List[Dict[str, Any]] = []
        for item in scan_result["to_clean"]:
            run_id = item.get("run_id")
            if not run_id:
                continue
            if dry_run:
                cleaned.append(run_id)
                continue
            run_dir = self._run_dir(run_id)
            try:
                import shutil
                shutil.rmtree(run_dir)
                cleaned.append(run_id)
            except Exception as e:
                errors.append({"run_id": run_id, "error": str(e)})
        return {
            "cleaned": cleaned,
            "errors": errors,
            "dry_run": dry_run,
            "total_scanned": len(scan_result["to_clean"]) + len(scan_result["to_keep"]),
        }
