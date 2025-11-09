#!/usr/bin/env python3

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
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("PiscesLx.Core.Observability.Reporter")

class PiscesLxCoreReporter:
    """A class for generating markdown and lightweight HTML reports in a specified directory.

    Attributes:
        reports_dir (str): The directory where reports will be generated.
    """

    def __init__(self, reports_dir: str) -> None:
        """Initialize the reporter with a directory to store reports.

        Args:
            reports_dir (str): The directory path where reports will be stored.
        """
        self.reports_dir = reports_dir
        try:
            os.makedirs(self.reports_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create reports directory {self.reports_dir}: {str(e)}")

    def write_device_report(self, data: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """Generate device reports in both markdown and HTML formats.

        Args:
            data (Dict[str, Any]): Device data to be included in the report.
            session_id (Optional[str]): ID of the session. Defaults to None.

        Returns:
            str: Path to the generated markdown file.
        """
        ts = self._ts()
        base = f"device-{session_id or 'session'}-{ts}"
        md_path = os.path.join(self.reports_dir, base + ".md")
        html_path = os.path.join(self.reports_dir, base + ".html")

        md = self._render_device_md(data, ts, session_id)
        self._write_text(md_path, md)
        self._write_text(html_path, self._wrap_html(md))
        return md_path

    def write_session_report(self, data: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """Generate session reports in both markdown and HTML formats.

        Args:
            data (Dict[str, Any]): Session data to be included in the report.
            session_id (Optional[str]): ID of the session. Defaults to None.

        Returns:
            str: Path to the generated markdown file.
        """
        ts = self._ts()
        base = f"session-{session_id or 'session'}-{ts}"
        md_path = os.path.join(self.reports_dir, base + ".md")
        html_path = os.path.join(self.reports_dir, base + ".html")

        md = self._render_session_md(data, ts, session_id)
        self._write_text(md_path, md)
        self._write_text(html_path, self._wrap_html(md))
        return md_path

    def write_release_notes(self, version: str, items: List[str], delta_since_prev: List[str]) -> str:
        """Generate a dedicated release notes report in the specified reports directory.

        Args:
            version (str): Project version string (e.g., 1.0.0150).
            items (List[str]): Changelog items of the specified version.
            delta_since_prev (List[str]): Optional delta items since the previous version.

        Returns:
            str: Path to the generated markdown file.
        """
        ts = self._ts()
        base = f"release-{version}-{ts}"
        md_path = os.path.join(self.reports_dir, base + ".md")
        html_path = os.path.join(self.reports_dir, base + ".html")

        lines: List[str] = [
            f"# Release Notes - Version {version}",
            "",
            f"- Generated: {ts}",
            "",
            "## Changes",
        ]
        for it in items:
            lines.append(f"- {it}")
        if delta_since_prev:
            lines.append("")
            lines.append("## Changes since previous version")
            for it in delta_since_prev:
                lines.append(f"- {it}")

        md = "\n".join(lines)
        self._write_text(md_path, md)
        self._write_text(html_path, self._wrap_html(md))
        return md_path

    def _render_device_md(self, data: Dict[str, Any], ts: str, session_id: Optional[str]) -> str:
        """Render device data into markdown format.

        Args:
            data (Dict[str, Any]): Device data to be rendered.
            ts (str): Timestamp of report generation.
            session_id (Optional[str]): ID of the session.

        Returns:
            str: Markdown content of the device report.
        """
        lines = [
            f"# PiscesL1 Device Report",
            "",
            f"- Generated: {ts}",
            f"- Session: {session_id or '-'}",
            "",
            "## Summary",
        ]
        hs = data.get("health_snapshot") or {}
        pm = hs.get("metrics") or {}
        if pm:
            try:
                req = max(int(pm.get("request_count", 0)), 1)
                err = float(pm.get("error_count", 0)) / req
            except Exception:
                err = float(pm.get("error_rate", 0.0))
            for k in ("cpu_usage", "memory_usage", "throughput", "p95_latency", "p99_latency"):
                if k in pm:
                    lines.append(f"- **{k}**: {pm.get(k)}")
            lines.append(f"- **error_rate**: {err:.2%}")
            lines.append("")
        env = data.get("environment") or {}
        if env:
            lines.append("## Environment")
            for k, v in env.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        lines.append("## Raw Data")
        try:
            raw = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            raw = str(data)
        lines.append("```json")
        lines.append(raw)
        lines.append("```")
        return "\n".join(lines)

    def _render_session_md(self, data: Dict[str, Any], ts: str, session_id: Optional[str]) -> str:
        """Render session data into markdown format.

        Args:
            data (Dict[str, Any]): Session data to be rendered.
            ts (str): Timestamp of report generation.
            session_id (Optional[str]): ID of the session.

        Returns:
            str: Markdown content of the session report.
        """
        lines = [
            f"# PiscesL1 Session Report",
            "",
            f"- Generated: {ts}",
            f"- Session: {session_id or '-'}",
            "",
            "## Summary",
        ]
        for k, v in (data.get("session_summary") or {}).items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

        fhs = data.get("final_health_snapshot") or {}
        fpm = fhs.get("metrics") or {}
        if fpm:
            lines.append("## Final Metrics Snapshot")
            try:
                req = max(int(fpm.get("request_count", 0)), 1)
                err = float(fpm.get("error_count", 0)) / req
            except Exception:
                err = float(fpm.get("error_rate", 0.0))
            for k in ("cpu_usage", "memory_usage", "throughput", "p95_latency", "p99_latency"):
                if k in fpm:
                    lines.append(f"- **{k}**: {fpm.get(k)}")
            lines.append(f"- **error_rate**: {err:.2%}")
            lines.append("")

        version = data.get("version")
        if version:
            # UL changelog functionality removed - version info preserved
            lines.append(f"## Version Information")
            lines.append(f"- **Version**: {version}")
            lines.append("")

        if data.get("events"):
            lines.append("## Events")
            for e in data["events"]:
                lines.append(f"- **{e.get('name','event')}**: {e.get('detail','')}")
            lines.append("")

        pm = data.get("performance_metrics")
        if isinstance(pm, dict):
            lines.append("## Metrics Summary")
            avg = pm.get("avg_latency", 0)
            p95 = pm.get("p95_latency", 0)
            p99 = pm.get("p99_latency", 0)
            thr = pm.get("throughput", 0)
            err = 0.0
            try:
                req = max(pm.get("request_count", 0), 1)
                err = float(pm.get("error_count", 0)) / req
            except Exception:
                err = 0.0
            lines.append(f"- **avg_latency**: {avg:.2f} ms")
            lines.append(f"- **p95_latency**: {p95:.2f} ms")
            lines.append(f"- **p99_latency**: {p99:.2f} ms")
            lines.append(f"- **throughput**: {thr:.2f} req/s")
            lines.append(f"- **error_rate**: {err:.2%}")
            lines.append("")

        try:
            from .service import PiscesLxCoreObservabilityService  # local import to avoid cycles
            reg = PiscesLxCoreObservabilityService.instance().metrics_registry()
            snap = reg.snapshot()
            gauges = snap.get("gauges", {})
            gpu_avg = gauges.get("gpu.util_percent.avg")
            gpu_keys = [k for k in gauges.keys() if k.startswith("gpu.") and k.endswith(".mem_total_mib")]
            if gpu_avg is not None or gpu_keys:
                lines.append("## GPU Summary")
                if gpu_avg is not None:
                    try:
                        lines.append(f"- **util_avg**: {float(gpu_avg):.2f} %")
                    except Exception:
                        lines.append(f"- **util_avg**: {gpu_avg}")
                shown = 0
                for k in sorted(gpu_keys)[:8]:
                    try:
                        idx = k.split(".")[1]
                    except Exception:
                        idx = "?"
                    total = gauges.get(k)
                    used = gauges.get(k.replace("mem_total_mib", "mem_used_mib"))
                    try:
                        line = f"- GPU {idx}: used {float(used):.0f} / {float(total):.0f} MiB"
                    except Exception:
                        line = f"- GPU {idx}: used {used} / {total} MiB"
                    lines.append(line)
                    shown += 1
                lines.append("")
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {str(e)}")

        anomalies = data.get("anomaly_detection")
        if isinstance(anomalies, list) and anomalies:
            lines.append("## Alerts")
            for a in anomalies:
                try:
                    lines.append(f"- [{a.get('severity','info')}] {a.get('pattern_type','anomaly')}: {a.get('description','')} (conf={a.get('confidence',0):.2f})")
                except Exception:
                    lines.append(f"- anomaly: {str(a)}")
            lines.append("")

        lines.append("## Raw Data")
        try:
            raw = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            raw = str(data)
        lines.append("```json")
        lines.append(raw)
        lines.append("```")
        return "\n".join(lines)

    def _write_text(self, path: str, text: str) -> None:
        """Write text content to a file.

        Args:
            path (str): Path to the file.
            text (str): Text content to be written.
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Failed to write text to file {path}: {str(e)}")

    def _wrap_html(self, md_text: str) -> str:
        """Wrap markdown text in a lightweight HTML format.

        Args:
            md_text (str): Markdown text to be wrapped.

        Returns:
            str: HTML content with wrapped markdown text.
        """
        safe = md_text.replace("<", "&lt;").replace(">", "&gt;")
        return (
            "<!doctype html><html><head><meta charset='utf-8'><title>Pisces Report"
            "</title><style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;}pre{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto;}</style></head>"
            f"<body><pre>{safe}</pre></body></html>"
        )

    def _ts(self) -> str:
        """Generate a UTC timestamp string.

        Returns:
            str: Formatted timestamp string in the format 'YYYYMMDD-HHMMSS'.
        """
        return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    def _read_ul_changelog(self, version: str) -> List[str]:
        """Attempt to read UL changelog items for a specified version.

        Returns an empty list when the changelog is unavailable. The path can be provided by the environment variable PISCES_UL_CHANGELOG_PATH.
        Supported formats: plain text (one item per line) or JSON list.

        Args:
            version (str): Project version string.

        Returns:
            List[str]: List of changelog items.
        """
        try:
            path = os.environ.get("PISCES_UL_CHANGELOG_PATH")
            if not path:
                return []
            if not os.path.exists(path):
                return []
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                try:
                    import json as _json
                    data = _json.loads(content)
                    if isinstance(data, list):
                        return [str(x) for x in data]
                except Exception as e:
                    logger.debug(f"Failed to parse changelog as JSON: {str(e)}")
                items = [ln.strip() for ln in content.splitlines() if ln.strip()]
                return items[:200]
            except Exception as e:
                logger.error(f"Failed to read changelog from {path}: {str(e)}")
                return []
        except Exception as e:
            logger.error(f"Unexpected error while reading UL changelog: {str(e)}")
            return []

