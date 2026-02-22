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

import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils.dc import PiscesLxLogger

from opss.run import (
    POPSSRunController,
    POPSSRunStore,
    POPSSResourceMonitor,
    POPSSHeartbeatMonitor,
)


class PiscesLxRunCancelled(Exception):
    pass


class PiscesLxTrainingRunReporter:
    def __init__(
        self,
        run_id: str,
        run_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        control_interval_s: float = 0.5,
        resource_interval_s: float = 5.0,
        heartbeat_interval_s: float = 10.0,
        heartbeat_timeout_s: float = 60.0,
        enable_resource_monitor: bool = True,
        enable_heartbeat: bool = True,
        enable_gpu_monitor: bool = True,
    ):
        self._logger = PiscesLxLogger("PiscesLxTrainingRunReporter")
        self._store = POPSSRunStore(run_id, run_dir=run_dir)
        self._ctl = POPSSRunController(self._store)
        self._control_interval_s = float(control_interval_s or 0.5)
        self._run_name = str(run_name).strip() if run_name is not None else ""
        self._paused = False
        self._cancel_requested = False
        self._trainer = None
        self._train_config = None
        self._resource_monitor: Optional[POPSSResourceMonitor] = None
        self._heartbeat_monitor: Optional[POPSSHeartbeatMonitor] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_running = False
        self._enable_resource_monitor = bool(enable_resource_monitor)
        self._enable_heartbeat = bool(enable_heartbeat)
        self._enable_gpu_monitor = bool(enable_gpu_monitor)
        self._resource_interval_s = float(resource_interval_s or 5.0)
        self._heartbeat_interval_s = float(heartbeat_interval_s or 10.0)
        self._heartbeat_timeout_s = float(heartbeat_timeout_s or 60.0)

    @property
    def store(self) -> POPSSRunStore:
        return self._store

    @property
    def controller(self) -> POPSSRunController:
        return self._ctl

    @property
    def resource_monitor(self) -> Optional[POPSSResourceMonitor]:
        return self._resource_monitor

    @property
    def heartbeat_monitor(self) -> Optional[POPSSHeartbeatMonitor]:
        return self._heartbeat_monitor

    def init(self, spec: Dict[str, Any]) -> None:
        s = dict(spec or {})
        s.setdefault("run_id", self._store.run_id)
        if self._run_name:
            s.setdefault("run_name", self._run_name)
        s.setdefault("type", "train")
        self._ctl.init_run(s, state={"status": "running", "phase": "initializing", "pid": int(os.getpid())})
        if self._enable_resource_monitor:
            self._resource_monitor = POPSSResourceMonitor(
                self._store,
                interval_s=self._resource_interval_s,
                enable_gpu=self._enable_gpu_monitor,
                enable_io=True,
            )
            self._resource_monitor.start()
            self._logger.info("Resource monitor started")
        if self._enable_heartbeat:
            self._heartbeat_monitor = POPSSHeartbeatMonitor(
                self._store,
                interval_s=self._heartbeat_interval_s,
                timeout_s=self._heartbeat_timeout_s,
            )
            self._start_heartbeat_loop()
            self._logger.info("Heartbeat monitor started")

    def _start_heartbeat_loop(self) -> None:
        if self._heartbeat_running:
            return
        self._heartbeat_running = True
        def _loop():
            while self._heartbeat_running:
                try:
                    if self._heartbeat_monitor is not None:
                        self._heartbeat_monitor.beat(metadata={"phase": "training"})
                except Exception:
                    pass
                time.sleep(self._heartbeat_interval_s)
        self._heartbeat_thread = threading.Thread(target=_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat_loop(self) -> None:
        self._heartbeat_running = False
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None

    def bind(self, trainer: Any, train_config: Any) -> None:
        self._trainer = trainer
        self._train_config = train_config

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _safe_float(self, v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    def _safe_int(self, v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            return None

    def _snapshot(self, patch: Dict[str, Any]) -> None:
        self._ctl.update_state(patch or {})

    def _emit_metric(self, metric: Dict[str, Any]) -> None:
        self._ctl.append_metric(metric or {})

    def _emit_event(self, t: str, payload: Optional[Dict[str, Any]] = None, level: str = "info") -> None:
        self._ctl.append_event(t, level=level, payload=payload or {})

    def _load_last_control_seq(self) -> int:
        st = self._store.read_state() or {}
        return int(st.get("last_control_seq") or 0)

    def _save_last_control_seq(self, seq: int) -> None:
        self._snapshot({"last_control_seq": int(seq)})

    def _apply_control(self, obj: Dict[str, Any]) -> None:
        action = str(obj.get("action") or "").strip()
        payload = obj.get("payload") or {}
        seq = obj.get("seq")

        if action == "pause":
            self._paused = True
            self._emit_event("control_applied", payload={"action": "pause", "seq": seq})
            self._emit_event("training_pause_requested", payload={"seq": seq})
            self._snapshot({"status": "paused", "phase": "paused"})
            return
        if action == "resume":
            self._paused = False
            self._emit_event("control_applied", payload={"action": "resume", "seq": seq})
            self._emit_event("training_resume_requested", payload={"seq": seq})
            self._snapshot({"status": "running", "phase": "training"})
            return
        if action == "cancel":
            self._cancel_requested = True
            self._emit_event("control_applied", payload={"action": "cancel", "seq": seq})
            self._emit_event("training_cancel_requested", payload={"seq": seq})
            self._snapshot({"status": "cancelled", "phase": "cancelling"})
            try:
                self._save_checkpoint_now(reason="cancel")
            except Exception:
                pass
            return
        if action == "save_ckpt_now":
            self._emit_event("control_applied", payload={"action": "save_ckpt_now", "seq": seq})
            self._emit_event("training_save_ckpt_now", payload={"seq": seq})
            self._save_checkpoint_now(reason="manual")
            return
        if action:
            self._emit_event("control_ignored", payload={"action": action, "seq": seq})

    def poll_controls(self) -> None:
        last = self._load_last_control_seq()
        new_last, items = self._ctl.poll_controls(last)
        if new_last != last:
            self._save_last_control_seq(new_last)
        for obj in items:
            self._apply_control(obj)

    def _pause_loop(self) -> None:
        while self._paused and not self._cancel_requested:
            time.sleep(self._control_interval_s)
            self.poll_controls()

    def _save_checkpoint_now(self, reason: str = "manual") -> Optional[str]:
        if self._trainer is None or self._train_config is None:
            return None
        out_dir = getattr(self._train_config, "output_dir", None) or ".pisceslx/checkpoints"
        os.makedirs(str(out_dir), exist_ok=True)
        step = getattr(self._trainer, "global_step", None)
        step_i = self._safe_int(step) or 0
        path = str(Path(out_dir) / f"checkpoint_step_{step_i}_{reason}.pt")
        try:
            self._trainer.save_checkpoint(path, metadata={"reason": str(reason), "global_step": step_i})
        except TypeError:
            self._trainer.save_checkpoint(path)
        self._ctl.record_artifact(f"checkpoint_{step_i}_{reason}", path)
        self._emit_event("checkpoint_saved", payload={"path": path, "step": step_i, "reason": reason})
        return path

    def get_resource_summary(self) -> Dict[str, Any]:
        if self._resource_monitor is None:
            return {"enabled": False}
        return self._resource_monitor.get_summary()

    def on_stage(self, stage: str, **kwargs) -> None:
        st = str(stage)
        if st == "training_start":
            self._snapshot({"status": "running", "phase": "training"})
            self._emit_event("training_start", payload={"meta": kwargs})
            return

        if st == "training_end":
            cur = self._store.read_state() or {}
            cur_status = str(cur.get("status") or "")
            if cur_status not in ("cancelled", "failed"):
                self._snapshot({"status": "completed", "phase": "completed"})
            self._emit_event("training_end", payload={"meta": kwargs})
            self._stop_heartbeat_loop()
            if self._resource_monitor is not None:
                self._resource_monitor.stop()
                resource_summary = self._resource_monitor.get_summary()
                self._emit_event("resource_summary", payload=resource_summary)
            return

        if st == "epoch_start":
            epoch = self._safe_int(kwargs.get("epoch"))
            if epoch is not None:
                self._snapshot({"epoch": epoch, "phase": "training"})
            self.poll_controls()
            if self._cancel_requested:
                raise PiscesLxRunCancelled("cancel_requested")
            if self._paused:
                self._pause_loop()
            return

        if st == "batch_end":
            step_result = kwargs.get("step_result") or {}
            try:
                step = self._safe_int(getattr(self._trainer, "global_step", None))
            except Exception:
                step = None
            loss = self._safe_float(step_result.get("loss"))
            lr = self._safe_float(step_result.get("learning_rate"))
            grad_norm = self._safe_float(step_result.get("grad_norm"))
            throughput = self._safe_float(step_result.get("throughput"))

            metric: Dict[str, Any] = {"type": "training"}
            if step is not None:
                metric["step"] = step
            if loss is not None:
                metric["loss"] = loss
            if lr is not None:
                metric["lr"] = lr
            if grad_norm is not None:
                metric["grad_norm"] = grad_norm
            if throughput is not None:
                metric["samples_per_s"] = throughput
            if self._resource_monitor is not None:
                snapshot = self._resource_monitor.last_snapshot
                if snapshot is not None:
                    metric["cpu_percent"] = snapshot.cpu_percent
                    metric["memory_mb"] = snapshot.memory_mb
                    if snapshot.gpu_memory_mb is not None:
                        metric["gpu_memory_mb"] = snapshot.gpu_memory_mb
            if metric:
                self._emit_metric(metric)
                self._snapshot({**{k: v for k, v in metric.items() if k != "type"}, "phase": "training", "status": "running"})

            self.poll_controls()
            if self._cancel_requested:
                raise PiscesLxRunCancelled("cancel_requested")
            if self._paused:
                self._pause_loop()
            return

        if st == "checkpoint_saved":
            path = kwargs.get("path")
            step = kwargs.get("global_step")
            step_i = self._safe_int(step)
            if path:
                key = f"checkpoint_{step_i}" if step_i is not None else "checkpoint"
                self._ctl.record_artifact(key, str(path))
            self._emit_event("checkpoint_saved", payload={"path": path, "step": step_i})
            return

        if st == "evaluation_end":
            eval_stats = kwargs.get("eval_stats") or {}
            self._emit_event("evaluation_end", payload={"eval": eval_stats})
            try:
                self._emit_metric({"type": "eval", "eval_loss": self._safe_float(eval_stats.get("eval_loss"))})
            except Exception:
                pass
            self.poll_controls()
            if self._cancel_requested:
                raise PiscesLxRunCancelled("cancel_requested")
            if self._paused:
                self._pause_loop()
            return

    def shutdown(self) -> None:
        self._stop_heartbeat_loop()
        if self._resource_monitor is not None:
            self._resource_monitor.stop()
        self._emit_event("reporter_shutdown", payload={})
