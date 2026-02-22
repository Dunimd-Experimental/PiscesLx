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

import json
import time
from typing import Optional

from .store import POPSSRunStore


class POPSSRunAttacher:
    def __init__(self, store: POPSSRunStore):
        self._store = store
        self._events_off = 0
        self._metrics_off = 0

    def attach(self, poll_interval_s: float = 0.5, follow: bool = True) -> int:
        idle_loops = 0
        while True:
            ev_chunk, self._events_off = self._store.tail_jsonl("events", self._events_off)
            mt_chunk, self._metrics_off = self._store.tail_jsonl("metrics", self._metrics_off)

            emitted = False
            if ev_chunk:
                emitted = True
                for line in ev_chunk.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        print(line)
                        continue
                    lvl = obj.get("level", "info")
                    et = obj.get("type", "event")
                    ts = obj.get("ts", "")
                    payload = obj.get("payload", {})
                    print(f"[{ts}] {lvl} {et} {payload}")

            if mt_chunk:
                emitted = True
                for line in mt_chunk.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    ts = obj.get("ts", "")
                    step = obj.get("step")
                    loss = obj.get("loss")
                    lr = obj.get("lr") or obj.get("learning_rate")
                    tps = obj.get("tokens_per_s") or obj.get("samples_per_s")
                    fields = []
                    if step is not None:
                        fields.append(f"step={step}")
                    if loss is not None:
                        fields.append(f"loss={loss}")
                    if lr is not None:
                        fields.append(f"lr={lr}")
                    if tps is not None:
                        fields.append(f"tps={tps}")
                    if fields:
                        print(f"[{ts}] metric " + " ".join(fields))

            if not follow:
                break

            if emitted:
                idle_loops = 0
            else:
                idle_loops += 1

            state = self._store.read_state()
            status = (state or {}).get("status")
            if status in ("completed", "failed", "cancelled", "stopped"):
                if idle_loops >= int(2.0 / max(poll_interval_s, 1e-3)):
                    return 0

            time.sleep(poll_interval_s)
        return 0

