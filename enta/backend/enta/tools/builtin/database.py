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

import json
import sqlite3
from typing import Any

from enta.tools.base import build_tool


async def _database_execute(**kwargs: Any) -> str:
    sql = kwargs.get("sql", "")
    database_url = kwargs.get("database_url", ":memory:")
    limit = kwargs.get("limit", 100)

    try:
        conn = sqlite3.connect(database_url)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)

        sql_upper = sql.strip().upper()
        if sql_upper.startswith("SELECT") or sql_upper.startswith("PRAGMA"):
            rows = cursor.fetchmany(limit)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = [dict(zip(columns, row, strict=False)) for row in rows]
            conn.commit()
            conn.close()
            result = {"columns": columns, "rows": results, "count": len(results)}
            if len(results) == limit:
                result["truncated"] = True
            return json.dumps(result, indent=2, default=str)
        else:
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            return json.dumps({"affected_rows": affected})

    except Exception as e:
        return f"Error executing SQL: {e}"


EncreDatabaseTool = build_tool(
    name="database",
    description="Execute SQL queries against connected databases",
    input_schema={
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The SQL query to execute",
            },
            "database_url": {
                "type": "string",
                "description": "Database connection URL (uses in-memory sqlite3 if omitted)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of result rows (default: 100)",
            },
        },
        "required": ["sql"],
    },
    execute=_database_execute,
    intents=["coding", "data"],
    is_concurrency_safe=lambda _: True,
)
