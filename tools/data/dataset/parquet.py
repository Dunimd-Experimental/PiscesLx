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

"""
Materialize Hub parquet shards into the JSONL corpora the trainer reads.

``PiscesLxTrainOrchestrator`` resolves a ``data.datasets`` entry either to a
``kaggle://`` locator or to a directory of ``.json`` / ``.jsonl`` files, while the
Hub downloaders deliver ``.parquet``.  This module is the missing link: it turns the
downloaded shards into one JSON object per record, keeping the fields the training
and alignment stages need and dropping empty or truncated documents.
"""

import argparse
import json
import pathlib
import sys

from utils.dc import PiscesLxLogger
from utils.paths import PiscesLxPaths

__all__ = ["PiscesLxToolsDataParquetCorpus"]


class PiscesLxToolsDataParquetCorpus:
    """Parquet shard to JSONL corpus conversion for the data stage."""

    TEXT_FIELDS = ("text", "content", "sentence", "question")
    PASSTHROUGH_FIELDS = ("title", "id", "answer", "output", "topic", "url")
    MIN_TEXT_CHARS = 1
    PROGRESS_RECORDS = 5000

    def __init__(self, source_dir: str, target_dir: str) -> None:
        self.source_dir = pathlib.Path(source_dir)
        self.target_dir = pathlib.Path(target_dir)
        self._LOG = PiscesLxLogger(
            "PiscesLx.Tools.Data.Parquet",
            file_path=PiscesLxPaths.get_log_file("PiscesLx.Tools.Data.Parquet"),
            enable_file=True,
        )

    def convert(self, pattern: str = "*.parquet", max_records: int = 0) -> list:
        """Convert every shard matching *pattern*; returns one report per shard."""
        import pyarrow.parquet as pq

        self.target_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        for shard in sorted(self.source_dir.glob(pattern)):
            # Row groups, not the whole table: the largest pretraining shards are
            # hundreds of megabytes of parquet and expand several times in memory.
            handle = pq.ParquetFile(shard)
            columns = handle.schema_arrow.names
            report = self._write_records(shard, columns, handle.iter_batches(), max_records)
            reports.append(report)
            self._LOG.info(
                f"{report['name']}: kept {report['records']} of {report['rows']} rows "
                f"({report['output_bytes']} bytes)"
            )
        return reports

    def _write_records(self, shard: pathlib.Path, columns, batches, max_records: int) -> dict:
        text_field = next((f for f in self.TEXT_FIELDS if f in columns), None)
        if text_field is None:
            raise ValueError(
                f"{shard.name}: none of {self.TEXT_FIELDS} is a column of {columns}"
            )
        carry = [f for f in self.PASSTHROUGH_FIELDS if f in columns]
        target = self.target_dir / f"{shard.stem}.jsonl"
        rows, kept, written_bytes, stop = 0, 0, 0, False
        with target.open("w", encoding="utf-8") as handle:
            for batch in batches:
                if stop:
                    break
                for row in batch.to_pylist():
                    rows += 1
                    if max_records and kept >= max_records:
                        stop = True
                        break
                    text = self._clean(row.get(text_field))
                    if len(text) < self.MIN_TEXT_CHARS:
                        continue
                    record = {text_field: text}
                    record.update({f: row[f] for f in carry if row.get(f) is not None})
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    handle.write(line)
                    written_bytes += len(line.encode("utf-8"))
                    kept += 1
                    if kept % self.PROGRESS_RECORDS == 0:
                        self._LOG.info(f"{shard.name}: {kept} records written")
        return {
            "name": shard.name,
            "output": target.as_posix(),
            "rows": rows,
            "records": kept,
            "output_bytes": written_bytes,
            "text_field": text_field,
        }

    @staticmethod
    def _clean(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            value = " ".join(str(part) for part in value if part)
        return str(value).strip()

    @classmethod
    def cli(cls, argv) -> int:
        """``--source`` / ``--target`` / ``--pattern`` / ``--max-records`` driver."""
        parser = argparse.ArgumentParser(prog="tools/data/dataset/parquet.py")
        parser.add_argument("--source", required=True, help="directory holding parquet shards")
        parser.add_argument("--target", required=True, help="directory to write JSONL into")
        parser.add_argument("--pattern", default="*.parquet")
        parser.add_argument("--max-records", type=int, default=0,
                            help="cap per shard; 0 keeps everything")
        args = parser.parse_args(argv)
        reports = cls(args.source, args.target).convert(args.pattern, args.max_records)
        for report in reports:
            print(json.dumps(report, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(PiscesLxToolsDataParquetCorpus.cli(sys.argv[1:]))
