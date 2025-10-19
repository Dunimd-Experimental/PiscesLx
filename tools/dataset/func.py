#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import ast
import json
import math
import random
import calendar
import statistics
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timezone, timedelta

_ALLOWED_BUILTINS = {
    'len': len,
    'min': min,
    'max': max,
    'sum': sum,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'abs': abs,
    'round': round,
    'sorted': sorted,
    'set': set,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'any': any,
    'all': all,
    'range': range,
}

_ALLOWED_MODULES = {
    're': re,
    'json': json,
}

_ALLOWED_AST_NODES = (
    # core
    ast.Module, ast.Expression, ast.Expr,
    ast.Constant, ast.Name, ast.Load,
    # structures
    ast.Dict, ast.List, ast.Tuple, ast.Set,
    ast.Subscript, ast.Slice, ast.Attribute,
    ast.Call,
    # expressions
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    # comps
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.comprehension,
    # operators: boolean
    ast.And, ast.Or, ast.Not,
    # operators: arithmetic
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    # operators: unary
    ast.UAdd, ast.USub, ast.Invert,
    # operators: comparisons
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
)

_DEFUSED_ATTR_PREFIX = "_"

class SafeValidator(ast.NodeVisitor):
    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed node: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith(_DEFUSED_ATTR_PREFIX):
            raise ValueError("Access to private/dunder attributes is not allowed")
        self.visit(node.value)

    def visit_Name(self, node: ast.Name):
        if node.id.startswith(_DEFUSED_ATTR_PREFIX):
            raise ValueError("Access to private/dunder names is not allowed")

    def visit_Call(self, node: ast.Call):
        self.visit(node.func)
        for a in node.args:
            self.visit(a)
        for kw in node.keywords:
            self.visit(kw.value)

# ---------------- Helpers for records ----------------

def get(rec: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested value by dotted path, supports a[0] indices."""
    cur: Any = rec
    try:
        tokens = re.split(r"\.(?![^\[]*\])", path) if path else []
        for tok in tokens:
            if not tok:
                continue
            m = re.match(r"^(\w+)(\[(\d+)\])?$", tok)
            if not m:
                return default
            key = m.group(1)
            idx = m.group(3)
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
            if idx is not None:
                i = int(idx)
                if isinstance(cur, list) and 0 <= i < len(cur):
                    cur = cur[i]
                else:
                    return default
        return cur
    except Exception:
        return default

def join_text(*parts: Any, sep: str = "") -> str:
    return sep.join(str(p) for p in parts if p is not None)

# -------- Text helpers --------
def lower(s: Any) -> str:
    return str(s).lower()

def upper(s: Any) -> str:
    return str(s).upper()

def strip(s: Any) -> str:
    return str(s).strip()

def replace(s: Any, old: str, new: str) -> str:
    return str(s).replace(old, new)

def split(s: Any, sep: str | None = None) -> list[str]:
    return str(s).split(sep)

def join_list(items: Any, sep: str = "") -> str:
    try:
        return sep.join(str(x) for x in items)
    except Exception:
        return str(items)

def contains(s: Any, sub: str) -> bool:
    return sub in str(s)

def startswith(s: Any, prefix: str) -> bool:
    return str(s).startswith(prefix)

def endswith(s: Any, suffix: str) -> bool:
    return str(s).endswith(suffix)

def truncate(s: Any, length: int, ellipsis: str = "…") -> str:
    text = str(s)
    if length < 0:
        return text
    return text if len(text) <= length else text[:max(0, length)] + ellipsis

# -------- JSON helpers --------
def json_loads(s: Any, default: Any = None) -> Any:
    try:
        return json.loads(str(s))
    except Exception:
        return default

def json_dumps(obj: Any, ensure_ascii: bool = False) -> str:
    try:
        return json.dumps(obj, ensure_ascii=ensure_ascii)
    except Exception:
        return str(obj)

# -------- List/collection helpers --------
def flatten(lst: Any) -> list[Any]:
    if not isinstance(lst, list):
        return [lst]
    out: list[Any] = []
    for x in lst:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out

# -------- Numeric helpers --------
def clamp(v: Any, lo: Any | None = None, hi: Any | None = None) -> Any:
    try:
        x = v
        if lo is not None and x < lo:
            x = lo
        if hi is not None and x > hi:
            x = hi
        return x
    except Exception:
        return v

# -------- Regex helpers --------
def regex_search(s: Any, pattern: str, flags: int = 0, group: int | str = 0, default: Any = None) -> Any:
    try:
        m = re.search(pattern, str(s), flags)
        if not m:
            return default
        try:
            return m.group(group)
        except Exception:
            return m.group(0)
    except Exception:
        return default

def regex_findall(s: Any, pattern: str, flags: int = 0) -> list[Any]:
    try:
        return re.findall(pattern, str(s), flags)
    except Exception:
        return []

def regex_sub(s: Any, pattern: str, repl: str, count: int = 0, flags: int = 0) -> str:
    try:
        return re.sub(pattern, repl, str(s), count=count, flags=flags)
    except Exception:
        return str(s)

# -------- Datetime helpers --------
_COMMON_DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
]

def date_parse(val: Any, fmt: str | None = None, default: Any = None) -> datetime | Any:
    try:
        if isinstance(val, datetime):
            return val
        s = str(val).strip()
        if not s:
            return default
        if fmt:
            return datetime.strptime(s, fmt)
        # try ISO
        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass
        # try common formats
        for f in _COMMON_DT_FORMATS:
            try:
                return datetime.strptime(s, f)
            except Exception:
                continue
        return default
    except Exception:
        return default

def date_format(val: Any, fmt: str = "%Y-%m-%d %H:%M:%S", default: str = "") -> str:
    try:
        dt = date_parse(val)
        if not isinstance(dt, datetime):
            return default
        return dt.strftime(fmt)
    except Exception:
        return default

def to_timestamp(val: Any, tz_aware: bool = False) -> int | None:
    try:
        dt = date_parse(val)
        if not isinstance(dt, datetime):
            return None
        if tz_aware and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None

def date_diff(a: Any, b: Any, unit: str = 'seconds') -> float | None:
    try:
        da = date_parse(a)
        db = date_parse(b)
        if not isinstance(da, datetime) or not isinstance(db, datetime):
            return None
        delta = da - db
        if unit == 'seconds':
            return delta.total_seconds()
        if unit == 'minutes':
            return delta.total_seconds() / 60.0
        if unit == 'hours':
            return delta.total_seconds() / 3600.0
        if unit == 'days':
            return delta.total_seconds() / 86400.0
        return delta.total_seconds()
    except Exception:
        return None

# -------- Dict lookup (VLOOKUP-like) --------
def dict_lookup(key: Any, mapping: Any, default: Any = None, case_insensitive: bool = False) -> Any:
    try:
        if isinstance(mapping, dict):
            if case_insensitive and isinstance(key, str):
                lk = str(key).lower()
                for k, v in mapping.items():
                    if isinstance(k, str) and k.lower() == lk:
                        return v
                return default
            return mapping.get(key, default)
        # list of pairs
        if isinstance(mapping, list):
            for item in mapping:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    k, v = item[0], item[1]
                    if case_insensitive and isinstance(k, str) and isinstance(key, str):
                        if k.lower() == key.lower():
                            return v
                    elif k == key:
                        return v
        return default
    except Exception:
        return default

# -------- Casting helpers --------
def to_int(x: Any, default: Any = None) -> int | Any:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return int(float(str(x)))
    except Exception:
        return default

def to_float(x: Any, default: Any = None) -> float | Any:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(str(x))
    except Exception:
        return default

def safe_number(x: Any, default: Any = None) -> float | int | Any:
    v = to_float(x, None)
    return v if v is not None else default

# ================= Excel-style Aliases =================
# ---- Text functions ----
def LEFT(text: Any, num_chars: int) -> str:
    s = str(text)
    n = max(0, int(num_chars))
    return s[:n]

def RIGHT(text: Any, num_chars: int) -> str:
    s = str(text)
    n = max(0, int(num_chars))
    return s[-n:] if n else ""

def MID(text: Any, start_num: int, num_chars: int) -> str:
    s = str(text)
    start = max(1, int(start_num)) - 1
    n = max(0, int(num_chars))
    return s[start:start+n]

def LEN(text: Any) -> int:
    return len(str(text))

def TRIM(text: Any) -> str:
    return strip(text)

def LOWER(text: Any) -> str:
    return lower(text)

def UPPER(text: Any) -> str:
    return upper(text)

def PROPER(text: Any) -> str:
    try:
        return str(text).title()
    except Exception:
        return str(text)

def SUBSTITUTE(text: Any, old_text: str, new_text: str, instance_num: int | None = None) -> str:
    s = str(text)
    if instance_num is None:
        return s.replace(old_text, new_text)
    # replace only specific occurrence
    try:
        i = int(instance_num)
        if i <= 0:
            return s
        parts = s.split(old_text)
        if len(parts) <= i:
            return s
        return old_text.join(parts[:i]) + new_text + old_text.join(parts[i:])
    except Exception:
        return s

def REPLACE(old_text: Any, start_num: int, num_chars: int, new_text: Any) -> str:
    s = str(old_text)
    start = max(1, int(start_num)) - 1
    n = max(0, int(num_chars))
    return s[:start] + str(new_text) + s[start+n:]

def FIND(find_text: str, within_text: Any, start_num: int = 1) -> int:
    s = str(within_text)
    i = max(1, int(start_num)) - 1
    pos = s.find(find_text, i)
    return 0 if pos < 0 else pos + 1

def SEARCH(find_text: str, within_text: Any, start_num: int = 1) -> int:
    return FIND(str(find_text).lower(), str(within_text).lower(), start_num)

def CONCAT(*args: Any) -> str:
    return "".join(str(a) for a in args)

def CONCATENATE(*args: Any) -> str:
    return CONCAT(*args)

def TEXTSPLIT(text: Any, delimiter: str) -> list[str]:
    return split(text, delimiter)

def TEXTAFTER(text: Any, delimiter: str, instance_num: int = 1) -> str:
    s = str(text)
    if instance_num <= 0:
        return s
    parts = s.split(delimiter)
    return delimiter.join(parts[instance_num:]) if len(parts) > instance_num else ""

def TEXTBEFORE(text: Any, delimiter: str, instance_num: int = 1) -> str:
    s = str(text)
    if instance_num <= 0:
        return ""
    parts = s.split(delimiter)
    return delimiter.join(parts[:instance_num-1]) if instance_num > 1 else parts[0] if parts else ""

def VALUE(text: Any) -> float | int | None:
    v = to_float(text, None)
    if v is None:
        return None
    # return int if integral
    return int(v) if abs(v - int(v)) < 1e-9 else v

def TEXT(value: Any, format_text: str | None = None) -> str:
    # Minimal support: if looks like Python datetime format, try; else str
    if format_text:
        dt = date_parse(value)
        if isinstance(dt, datetime):
            return date_format(dt, fmt=format_text)
    return str(value)

# ---- Logic functions ----
def IF(condition: Any, value_if_true: Any, value_if_false: Any) -> Any:
    return value_if_true if bool(condition) else value_if_false

def AND(*args: Any) -> bool:
    return all(bool(x) for x in args)

def OR(*args: Any) -> bool:
    return any(bool(x) for x in args)

def NOT(x: Any) -> bool:
    return not bool(x)

def IFERROR(value: Any, value_if_error: Any) -> Any:
    try:
        return value()
    except TypeError:
        # if value is not callable, just return it unless it's an error container
        return value if value is not None else value_if_error
    except Exception:
        return value_if_error

def IFNA(value: Any, value_if_na: Any) -> Any:
    return value if value is not None else value_if_na

# ---- Math/Stat functions ----
def SUM(*args: Any) -> float:
    total = 0.0
    for a in args:
        if isinstance(a, (list, tuple)):
            for x in a:
                v = to_float(x, None)
                if v is not None:
                    total += v
        else:
            v = to_float(a, None)
            if v is not None:
                total += v
    return total

def AVERAGE(*args: Any) -> float | None:
    vals = []
    for a in args:
        if isinstance(a, (list, tuple)):
            for x in a:
                v = to_float(x, None)
                if v is not None:
                    vals.append(v)
        else:
            v = to_float(a, None)
            if v is not None:
                vals.append(v)
    return statistics.mean(vals) if vals else None

def MEDIAN(*args: Any) -> float | None:
    vals = []
    for a in args:
        if isinstance(a, (list, tuple)):
            for x in a:
                v = to_float(x, None)
                if v is not None:
                    vals.append(v)
        else:
            v = to_float(a, None)
            if v is not None:
                vals.append(v)
    return statistics.median(vals) if vals else None

def MODE(*args: Any) -> float | None:
    vals = []
    for a in args:
        if isinstance(a, (list, tuple)):
            for x in a:
                v = to_float(x, None)
                if v is not None:
                    vals.append(v)
        else:
            v = to_float(a, None)
            if v is not None:
                vals.append(v)
    try:
        return statistics.mode(vals) if vals else None
    except Exception:
        return None

def ROUNDDOWN(number: Any, num_digits: int = 0) -> float:
    n = to_float(number, 0.0)
    d = int(num_digits)
    factor = 10 ** d
    return math.floor(n * factor) / factor

def ROUNDUP(number: Any, num_digits: int = 0) -> float:
    n = to_float(number, 0.0)
    d = int(num_digits)
    factor = 10 ** d
    return math.ceil(n * factor) / factor

def INT_(number: Any) -> int:
    # Excel INT: floor toward -inf
    n = to_float(number, 0.0)
    return math.floor(n)

def INT(number: Any) -> int:
    return INT_(number)

def MOD(number: Any, divisor: Any) -> float | None:
    a = to_float(number, None)
    b = to_float(divisor, None)
    if a is None or b in (None, 0):
        return None
    return a % b

def POWER(number: Any, power: Any) -> float | None:
    a = to_float(number, None)
    b = to_float(power, None)
    if a is None or b is None:
        return None
    return a ** b

def SQRT(number: Any) -> float | None:
    a = to_float(number, None)
    if a is None or a < 0:
        return None
    return math.sqrt(a)

def RAND() -> float:
    return random.random()

def RANDBETWEEN(bottom: Any, top: Any) -> int | None:
    lo = to_int(bottom, None)
    hi = to_int(top, None)
    if lo is None or hi is None or lo > hi:
        return None
    return random.randint(lo, hi)

# Helper: iterate over values inside possibly nested lists/tuples for COUNT-like funcs
def _iter_values(*args: Any):
    for a in args:
        if isinstance(a, (list, tuple)):
            for x in a:
                yield x
        else:
            yield a

# Excel ROUND (halves away from zero), CEILING, FLOOR
def ROUND(number: Any, num_digits: int = 0) -> float:
    n = to_float(number, 0.0)
    d = int(num_digits)
    factor = 10 ** d
    x = n * factor
    if x >= 0:
        y = math.floor(x + 0.5)
    else:
        y = -math.floor(-x + 0.5)
    return y / factor

def CEILING(number: Any, significance: Any = 1) -> float | None:
    n = to_float(number, None)
    s = to_float(significance, None)
    if n is None or s in (None, 0):
        return None
    return math.ceil(n / s) * s

def FLOOR(number: Any, significance: Any = 1) -> float | None:
    n = to_float(number, None)
    s = to_float(significance, None)
    if n is None or s in (None, 0):
        return None
    return math.floor(n / s) * s

# COUNT family
def COUNT(*args: Any) -> int:
    c = 0
    for v in _iter_values(*args):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            c += 1
        else:
            num = to_float(v, None)
            if num is not None:
                c += 1
    return c

def COUNTA(*args: Any) -> int:
    c = 0
    for v in _iter_values(*args):
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        c += 1
    return c

def COUNTBLANK(*args: Any) -> int:
    c = 0
    for v in _iter_values(*args):
        if v is None or (isinstance(v, str) and v == ""):
            c += 1
    return c

def MIN_(*args: Any) -> float | None:
    vals = [to_float(v, None) for v in _iter_values(*args)]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None

def MAX_(*args: Any) -> float | None:
    vals = [to_float(v, None) for v in _iter_values(*args)]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None

def SMALL(array: Any, k: int) -> float | None:
    vals = [to_float(v, None) for v in _iter_values(array)]
    vals = [v for v in vals if v is not None]
    vals.sort()
    ki = int(k) - 1
    if 0 <= ki < len(vals):
        return vals[ki]
    return None

def LARGE(array: Any, k: int) -> float | None:
    vals = [to_float(v, None) for v in _iter_values(array)]
    vals = [v for v in vals if v is not None]
    vals.sort(reverse=True)
    ki = int(k) - 1
    if 0 <= ki < len(vals):
        return vals[ki]
    return None

def TEXTJOIN(delimiter: str, ignore_empty: bool, array: Any) -> str:
    parts = []
    for v in _iter_values(array):
        if v is None:
            if not ignore_empty:
                parts.append("")
            continue
        s = str(v)
        if ignore_empty and s == "":
            continue
        parts.append(s)
    return delimiter.join(parts)

def UNIQUE(array: Any) -> list[Any]:
    seen = set()
    out = []
    for v in _iter_values(array):
        key = v
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out

def FILTER(array: Any, include: Any) -> list[Any]:
    if not isinstance(array, (list, tuple)) or not isinstance(include, (list, tuple)):
        return []
    out = []
    for v, inc in zip(array, include):
        if bool(inc):
            out.append(v)
    return out

def SORT(array: Any, descending: bool = False) -> list[Any]:
    if not isinstance(array, (list, tuple)):
        return []
    try:
        return sorted(list(array), reverse=bool(descending))
    except Exception:
        return list(array)

# ---- Array/Lookup helpers (additional) ----
def INDEX(array: Any, row_num: int, column_num: int | None = None) -> Any:
    try:
        r = int(row_num) - 1
        if column_num is None:
            if isinstance(array, (list, tuple)):
                return array[r] if 0 <= r < len(array) else None
            return None
        c = int(column_num) - 1
        if isinstance(array, (list, tuple)) and 0 <= r < len(array):
            row = array[r]
            if isinstance(row, (list, tuple)) and 0 <= c < len(row):
                return row[c]
        return None
    except Exception:
        return None

def MATCH(lookup_value: Any, lookup_array: Any, match_type: int = 0) -> int | None:
    try:
        if not isinstance(lookup_array, (list, tuple)):
            return None
        arr = list(lookup_array)
        # exact match
        if int(match_type) == 0:
            for i, v in enumerate(arr):
                if v == lookup_value:
                    return i + 1
            return None
        # approximate: assume sorted ascending for +1, descending for -1
        if int(match_type) > 0:
            best = None
            for i, v in enumerate(arr):
                try:
                    if v <= lookup_value:
                        best = i + 1
                    else:
                        break
                except Exception:
                    continue
            return best
        else:
            best = None
            for i, v in enumerate(arr):
                try:
                    if v >= lookup_value:
                        best = i + 1
                        break
                except Exception:
                    continue
            return best
    except Exception:
        return None

def _build_criteria_func(criteria: Any):
    # Supports: ">5", ">=10", "<3", "<=2", "<>x", "=x", "x" (equals), wildcard * and ? for text
    if criteria is None:
        return lambda x: x is None
    if isinstance(criteria, (int, float)):
        return lambda x: to_float(x, None) == float(criteria)
    s = str(criteria)
    ops = [">=", "<=", "<>", ">", "<", "="]
    op = None
    rhs = s
    for o in ops:
        if s.startswith(o):
            op = o
            rhs = s[len(o):]
            break
    # wildcard handling for text equality when no numeric comparison
    def wildcard_match(text: str, pattern: str) -> bool:
        # Convert Excel wildcards * ? to regex
        try:
            pat = '^' + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + '$'
            return re.match(pat, text) is not None
        except Exception:
            return False

    if op is None:
        # equals with wildcard for text
        val = rhs
        def f(x):
            xs = str(x)
            return wildcard_match(xs, str(val))
        return f

    # try numeric compare
    num = to_float(rhs, None)
    if num is not None:
        if op == ">=":
            return lambda x: (to_float(x, None) is not None) and to_float(x, None) >= num
        if op == ">":
            return lambda x: (to_float(x, None) is not None) and to_float(x, None) > num
        if op == "<=":
            return lambda x: (to_float(x, None) is not None) and to_float(x, None) <= num
        if op == "<":
            return lambda x: (to_float(x, None) is not None) and to_float(x, None) < num
        if op == "=":
            return lambda x: to_float(x, None) == num
        if op == "<>":
            return lambda x: to_float(x, None) != num
    # text compare
    if op == "=":
        return lambda x: str(x) == rhs
    if op == "<>":
        return lambda x: str(x) != rhs
    # unsupported operator on text: fallback False
    return lambda x: False

def COUNTIF(range_: Any, criteria: Any) -> int:
    if not isinstance(range_, (list, tuple)):
        range_ = [range_]
    pred = _build_criteria_func(criteria)
    c = 0
    for v in range_:
        if pred(v):
            c += 1
    return c

def SUMIF(range_: Any, criteria: Any, sum_range: Any | None = None) -> float:
    if not isinstance(range_, (list, tuple)):
        range_ = [range_]
    if sum_range is None:
        sum_range = range_
    elif not isinstance(sum_range, (list, tuple)):
        sum_range = [sum_range]
    pred = _build_criteria_func(criteria)
    total = 0.0
    for a, b in zip(range_, sum_range):
        if pred(a):
            v = to_float(b, None)
            if v is not None:
                total += v
    return total

def AVERAGEIF(range_: Any, criteria: Any, average_range: Any | None = None) -> float | None:
    if not isinstance(range_, (list, tuple)):
        range_ = [range_]
    if average_range is None:
        average_range = range_
    elif not isinstance(average_range, (list, tuple)):
        average_range = [average_range]
    pred = _build_criteria_func(criteria)
    vals = []
    for a, b in zip(range_, average_range):
        if pred(a):
            v = to_float(b, None)
            if v is not None:
                vals.append(v)
    return statistics.mean(vals) if vals else None

def COUNTIFS(*args: Any) -> int:
    # args: range1, criteria1, range2, criteria2, ...
    if len(args) % 2 != 0 or len(args) == 0:
        return 0
    ranges = []
    preds = []
    for i in range(0, len(args), 2):
        r = args[i]
        c = args[i+1]
        if not isinstance(r, (list, tuple)):
            r = [r]
        ranges.append(list(r))
        preds.append(_build_criteria_func(c))
    count = 0
    for idx in range(len(ranges[0])):
        ok = True
        for r, p in zip(ranges, preds):
            v = r[idx] if idx < len(r) else None
            if not p(v):
                ok = False
                break
        if ok:
            count += 1
    return count

def SUMIFS(sum_range: Any, *args: Any) -> float:
    if not isinstance(sum_range, (list, tuple)):
        sum_vals = [sum_range]
    else:
        sum_vals = list(sum_range)
    if len(args) % 2 != 0:
        return 0.0
    ranges = []
    preds = []
    for i in range(0, len(args), 2):
        r = args[i]
        c = args[i+1]
        if not isinstance(r, (list, tuple)):
            r = [r]
        ranges.append(list(r))
        preds.append(_build_criteria_func(c))
    total = 0.0
    for idx in range(len(sum_vals)):
        ok = True
        for r, p in zip(ranges, preds):
            v = r[idx] if idx < len(r) else None
            if not p(v):
                ok = False
                break
        if ok:
            v = to_float(sum_vals[idx] if idx < len(sum_vals) else None, None)
            if v is not None:
                total += v
    return total

def AVERAGEIFS(average_range: Any, *args: Any) -> float | None:
    if not isinstance(average_range, (list, tuple)):
        av = [average_range]
    else:
        av = list(average_range)
    if len(args) % 2 != 0:
        return None
    ranges = []
    preds = []
    for i in range(0, len(args), 2):
        r = args[i]
        c = args[i+1]
        if not isinstance(r, (list, tuple)):
            r = [r]
        ranges.append(list(r))
        preds.append(_build_criteria_func(c))
    vals = []
    for idx in range(len(av)):
        ok = True
        for r, p in zip(ranges, preds):
            v = r[idx] if idx < len(r) else None
            if not p(v):
                ok = False
                break
        if ok:
            v = to_float(av[idx] if idx < len(av) else None, None)
            if v is not None:
                vals.append(v)
    return statistics.mean(vals) if vals else None

# ---- Additional text and info functions ----
def EXACT(text1: Any, text2: Any) -> bool:
    return str(text1) == str(text2)

def CLEAN(text: Any) -> str:
    s = str(text)
    return ''.join(ch for ch in s if ord(ch) >= 32)

def REPT(text: Any, number_times: Any) -> str:
    try:
        n = max(0, int(number_times))
    except Exception:
        n = 0
    return str(text) * n

def CHAR(number: Any) -> str:
    try:
        n = int(number)
        if 0 <= n <= 255:
            return chr(n)
    except Exception:
        pass
    return ""

def CODE(text: Any) -> int | None:
    s = str(text)
    return ord(s[0]) if s else None

def ISBLANK(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")

def ISNUMBER(value: Any) -> bool:
    return to_float(value, None) is not None and not isinstance(value, bool)

def ISTEXT(value: Any) -> bool:
    return isinstance(value, str)

def ISLOGICAL(value: Any) -> bool:
    return isinstance(value, bool)

def ISNA(value: Any) -> bool:
    return value is None

def ISNONTEXT(value: Any) -> bool:
    return not isinstance(value, str)

def TRANSPOSE(array: Any) -> list[Any]:
    if not isinstance(array, (list, tuple)):
        return [[array]]
    if len(array) == 0:
        return []
    if all(not isinstance(row, (list, tuple)) for row in array):
        # 1D -> column vector
        return [[x] for x in array]
    # assume rectangular
    rows = len(array)
    cols = max(len(row) if isinstance(row, (list, tuple)) else 0 for row in array)
    out = []
    for c in range(cols):
        out_row = []
        for r in range(rows):
            row = array[r]
            if isinstance(row, (list, tuple)) and c < len(row):
                out_row.append(row[c])
            else:
                out_row.append(None)
        out.append(out_row)
    return out

def CHOOSE(index_num: int, *values: Any) -> Any:
    i = int(index_num) - 1
    if 0 <= i < len(values):
        return values[i]
    return None

# ---- HELP (Excel-style) ----
_HELP_TEXT: dict[str, str] = {
    # Text
    'LEFT': 'LEFT(text, num_chars) -> 截取左侧指定字符数',
    'RIGHT': 'RIGHT(text, num_chars) -> 截取右侧指定字符数',
    'MID': 'MID(text, start_num, num_chars) -> 从位置起截取指定长度',
    'LEN': 'LEN(text) -> 文本长度',
    'TRIM': 'TRIM(text) -> 去除首尾空白',
    'LOWER': 'LOWER(text) -> 转小写',
    'UPPER': 'UPPER(text) -> 转大写',
    'PROPER': 'PROPER(text) -> 单词首字母大写',
    'SUBSTITUTE': 'SUBSTITUTE(text, old_text, new_text, [instance_num]) -> 替换文本',
    'REPLACE': 'REPLACE(old_text, start_num, num_chars, new_text) -> 替换区间',
    'FIND': 'FIND(find_text, within_text, [start_num]) -> 区分大小写查找，返回位置(1基)',
    'SEARCH': 'SEARCH(find_text, within_text, [start_num]) -> 不区分大小写查找，返回位置(1基)',
    'CONCAT': 'CONCAT(value1, [value2], ...) -> 连接文本',
    'CONCATENATE': 'CONCATENATE(value1, [value2], ...) -> 连接文本(同CONCAT)',
    'TEXTSPLIT': 'TEXTSPLIT(text, delimiter) -> 按分隔符拆分',
    'TEXTAFTER': 'TEXTAFTER(text, delimiter, [instance_num]) -> 返回分隔符之后',
    'TEXTBEFORE': 'TEXTBEFORE(text, delimiter, [instance_num]) -> 返回分隔符之前',
    'VALUE': 'VALUE(text) -> 将文本数字转为数值',
    'TEXT': 'TEXT(value, [format_text]) -> 格式化为文本',
    'CLEAN': 'CLEAN(text) -> 移除不可打印字符',
    'EXACT': 'EXACT(text1, text2) -> 比较是否完全相同',
    'REPT': 'REPT(text, number_times) -> 重复文本',
    'CHAR': 'CHAR(number) -> ASCII字符',
    'CODE': 'CODE(text) -> 首字符的ASCII码',
    # Logic
    'IF': 'IF(condition, value_if_true, value_if_false) -> 条件返回',
    'AND': 'AND(logical1, [logical2], ...) -> 全为真',
    'OR': 'OR(logical1, [logical2], ...) -> 任一为真',
    'NOT': 'NOT(logical) -> 逻辑非',
    'IFERROR': 'IFERROR(value, value_if_error) -> 出错时返回备用值',
    'IFNA': 'IFNA(value, value_if_na) -> 空/None时返回备用值',
    # Math/Stat
    'SUM': 'SUM(number1, [number2], ...) -> 求和(支持数组)',
    'AVERAGE': 'AVERAGE(number1, [number2], ...) -> 平均值',
    'MEDIAN': 'MEDIAN(number1, ...) -> 中位数',
    'MODE': 'MODE(number1, ...) -> 众数',
    'ROUND': 'ROUND(number, [num_digits]) -> 四舍五入(0.5远离0)',
    'ROUNDDOWN': 'ROUNDDOWN(number, [num_digits]) -> 向下取整(位数)',
    'ROUNDUP': 'ROUNDUP(number, [num_digits]) -> 向上取整(位数)',
    'INT': 'INT(number) -> 向下取整(负数更小)',
    'MOD': 'MOD(number, divisor) -> 取余',
    'POWER': 'POWER(number, power) -> 幂',
    'SQRT': 'SQRT(number) -> 平方根',
    'RAND': 'RAND() -> [0,1) 随机数',
    'RANDBETWEEN': 'RANDBETWEEN(bottom, top) -> 整数随机数',
    'COUNT': 'COUNT(values) -> 计数(数值/可转数值)',
    'COUNTA': 'COUNTA(values) -> 非空计数',
    'COUNTBLANK': 'COUNTBLANK(values) -> 空白计数',
    'MIN': 'MIN(values) -> 最小值',
    'MAX': 'MAX(values) -> 最大值',
    'SMALL': 'SMALL(array, k) -> 第k小',
    'LARGE': 'LARGE(array, k) -> 第k大',
    'CEILING': 'CEILING(number, [significance]) -> 向上到倍数',
    'FLOOR': 'FLOOR(number, [significance]) -> 向下到倍数',
    # Date/Time
    'DATE': 'DATE(year, month, day) -> 构造日期',
    'TIME': 'TIME(hour, minute, second) -> 构造时间(基准日)',
    'TODAY': 'TODAY() -> 今天日期',
    'NOW': 'NOW() -> 当前日期时间',
    'YEAR': 'YEAR(date) -> 年',
    'MONTH': 'MONTH(date) -> 月',
    'DAY': 'DAY(date) -> 日',
    'HOUR': 'HOUR(datetime) -> 时',
    'MINUTE': 'MINUTE(datetime) -> 分',
    'SECOND': 'SECOND(datetime) -> 秒',
    'WEEKDAY': 'WEEKDAY(date, [return_type]) -> 星期序号',
    'EOMONTH': 'EOMONTH(start_date, months) -> 月末日期',
    # Lookup/Array
    'XLOOKUP': 'XLOOKUP(lookup_value, lookup_array, return_array, [if_not_found]) -> 精确查找',
    'VLOOKUP': 'VLOOKUP(lookup_value, table_array, col_index, [range_lookup], [if_not_found]) -> 纵向查找(简化)',
    'INDEX': 'INDEX(array, row_num, [column_num]) -> 索引取值(1基)',
    'MATCH': 'MATCH(lookup_value, lookup_array, [match_type]) -> 返回位置(1基)',
    'TEXTJOIN': 'TEXTJOIN(delimiter, ignore_empty, array) -> 连接数组文本',
    'UNIQUE': 'UNIQUE(array) -> 去重(保持顺序)',
    'FILTER': 'FILTER(array, include) -> 布尔筛选',
    'SORT': 'SORT(array, [descending]) -> 排序',
    'TRANSPOSE': 'TRANSPOSE(array) -> 行列转置',
    'CHOOSE': 'CHOOSE(index_num, value1, [value2], ...) -> 从列表中选择',
    # Criteria Aggregations
    'COUNTIF': 'COUNTIF(range, criteria) -> 条件计数(支持 >,>=,<,<=,=,<> 以及通配符*)',
    'SUMIF': 'SUMIF(range, criteria, [sum_range]) -> 条件求和',
    'AVERAGEIF': 'AVERAGEIF(range, criteria, [average_range]) -> 条件平均',
    'COUNTIFS': 'COUNTIFS(range1, criteria1, range2, criteria2, ...) -> 多条件计数',
    'SUMIFS': 'SUMIFS(sum_range, range1, criteria1, ...) -> 多条件求和',
    'AVERAGEIFS': 'AVERAGEIFS(average_range, range1, criteria1, ...) -> 多条件平均',
}

# 为下拉框准备的分组信息（UI 可直接使用）
_HELP_GROUPS: dict[str, list[str]] = {
    'Text': ['LEFT','RIGHT','MID','LEN','TRIM','LOWER','UPPER','PROPER','SUBSTITUTE','REPLACE','FIND','SEARCH','CONCAT','CONCATENATE','TEXTSPLIT','TEXTAFTER','TEXTBEFORE','VALUE','TEXT','CLEAN','EXACT','REPT','CHAR','CODE'],
    'Logic': ['IF','AND','OR','NOT','IFERROR','IFNA'],
    'Math/Stat': ['SUM','AVERAGE','MEDIAN','MODE','ROUND','ROUNDDOWN','ROUNDUP','INT','MOD','POWER','SQRT','RAND','RANDBETWEEN','COUNT','COUNTA','COUNTBLANK','MIN','MAX','SMALL','LARGE','CEILING','FLOOR'],
    'Date/Time': ['DATE','TIME','TODAY','NOW','YEAR','MONTH','DAY','HOUR','MINUTE','SECOND','WEEKDAY','EOMONTH'],
    'Lookup/Array': ['XLOOKUP','VLOOKUP','INDEX','MATCH','TEXTJOIN','UNIQUE','FILTER','SORT','TRANSPOSE','CHOOSE'],
    'Criteria': ['COUNTIF','SUMIF','AVERAGEIF','COUNTIFS','SUMIFS','AVERAGEIFS'],
}

def HELP(name: Any | None = None) -> Any:
    """
    Provide help information for functions.

    Args:
        name (Any | None, optional): 
            - None: Return a brief usage instruction (for viewing in expressions).
            - '*' or 'ALL': Return group options friendly to dropdown boxes: [{group,label,value,desc}].
            - 'GROUP:<Name>': Return a brief list of functions in the specified group (e.g., GROUP:Text).
            - '?keyword': Perform a fuzzy search for function names by keyword and return a brief list of matched entries.
            - Function name or function object: Return a brief usage description of the function (string), 
              case-insensitive, and aliases are supported.

    Returns:
        Any: Help information based on the input parameter.
    """
    try:
        # 1) No parameter: Return a concise description to avoid filling the UI.
        if name is None:
            return (
                "Usage: HELP(name) | HELP('*') | HELP('ALL') | HELP('GROUP:Text') | HELP('?kw')\n"
                "Examples: HELP(LEFT) / HELP('LEFT') / HELP('GROUP:Date/Time') / HELP('?look')"
            )

        # Standardize string parameters.
        if isinstance(name, str):
            s = name.strip()
        else:
            s = name

        # 2) Support passing function objects: HELP(LEFT).
        if callable(name):
            raw = getattr(name, '__name__', str(name))
        else:
            raw = str(s)

        key = raw.upper()
        # 3) ALL: Return a complete group list for UI use.
        if key in ("*", "ALL"):
            out: list[dict[str, str]] = []
            for group, names in _HELP_GROUPS.items():
                for n in names:
                    if n in _HELP_TEXT:
                        out.append({'group': group, 'label': n, 'value': n, 'desc': _HELP_TEXT[n]})
            return out

        # 4) GROUP:xxx -> Return a brief list of the group.
        if key.startswith("GROUP:"):
            gname = raw.split(":", 1)[1]
            # Try case-insensitive matching of group names.
            target = None
            for g in _HELP_GROUPS.keys():
                if g.lower() == gname.lower():
                    target = g
                    break
            if target is None:
                return f"Unknown group: {gname}"
            items = []
            for n in _HELP_GROUPS[target]:
                if n in _HELP_TEXT:
                    items.append(f"{n} - {_HELP_TEXT[n]}")
            return items

        # 5) Fuzzy search: ?keyword
        if isinstance(s, str) and s.startswith('?'):
            kw = s[1:].strip().lower()
            if not kw:
                return []
            hits = []
            for name_key, desc in _HELP_TEXT.items():
                if kw in name_key.lower():
                    hits.append(f"{name_key} - {desc}")
            return hits

        # 6) Exact function name
        if key in _HELP_TEXT:
            return _HELP_TEXT[key]
        # Simple alias mapping
        aliases = {
            'CONCATENATE': 'CONCAT',
        }
        if key in aliases and aliases[key] in _HELP_TEXT:
            return _HELP_TEXT[aliases[key]]
        return f"No built-in description: {key}"
    except Exception as e:
        return f"HELP error: {e}"

# ---- Date/Time Excel-like ----
def DATE(year: int, month: int, day: int) -> datetime:
    return datetime(int(year), int(month), int(day))

def TIME(h: int, m: int, s: int) -> datetime:
    return datetime(1970, 1, 1, int(h), int(m), int(s))

def TODAY() -> datetime:
    now = datetime.now()
    return datetime(now.year, now.month, now.day)

def NOW() -> datetime:
    return datetime.now()

def YEAR(val: Any) -> int | None:
    dt = date_parse(val)
    return dt.year if isinstance(dt, datetime) else None

def MONTH(val: Any) -> int | None:
    dt = date_parse(val)
    return dt.month if isinstance(dt, datetime) else None

def DAY(val: Any) -> int | None:
    dt = date_parse(val)
    return dt.day if isinstance(dt, datetime) else None

def HOUR(val: Any) -> int | None:
    dt = date_parse(val)
    return dt.hour if isinstance(dt, datetime) else None

def MINUTE(val: Any) -> int | None:
    dt = date_parse(val)
    return dt.minute if isinstance(dt, datetime) else None

def SECOND(val: Any) -> int | None:
    dt = date_parse(val)
    return dt.second if isinstance(dt, datetime) else None

def WEEKDAY(val: Any, return_type: int = 1) -> int | None:
    dt = date_parse(val)
    if not isinstance(dt, datetime):
        return None
    # Python weekday: Monday=0..Sunday=6
    wd = dt.weekday()
    if return_type == 1:
        return (wd + 1) % 7 + 1  # Sunday=1..Saturday=7
    if return_type == 2:
        return wd + 1            # Monday=1..Sunday=7
    if return_type == 3:
        return ((wd + 6) % 7) + 1  # Monday=0=>1, Sunday=6=>7
    return (wd + 1) % 7 + 1

def EOMONTH(start_date: Any, months: int) -> datetime | None:
    dt = date_parse(start_date)
    if not isinstance(dt, datetime):
        return None
    y = dt.year + (dt.month - 1 + int(months)) // 12
    m = (dt.month - 1 + int(months)) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    return datetime(y, m, last_day)

# ---- Lookup (simplified) ----
def XLOOKUP(lookup_value: Any, lookup_array: Any, return_array: Any, if_not_found: Any = None) -> Any:
    try:
        if isinstance(lookup_array, (list, tuple)) and isinstance(return_array, (list, tuple)) and len(lookup_array) == len(return_array):
            for k, v in zip(lookup_array, return_array):
                if k == lookup_value:
                    return v
        if isinstance(lookup_array, dict):
            return lookup_array.get(lookup_value, if_not_found)
        return if_not_found
    except Exception:
        return if_not_found

def VLOOKUP(lookup_value: Any, table_array: Any, col_index: int, range_lookup: bool = False, if_not_found: Any = None) -> Any:
    try:
        if not isinstance(table_array, (list, tuple)) or int(col_index) < 1:
            return if_not_found
        idx = int(col_index) - 1
        for row in table_array:
            if isinstance(row, (list, tuple)) and len(row) > idx and row[0] == lookup_value:
                return row[idx]
        return if_not_found
    except Exception:
        return if_not_found

def chat_to_qa(rec: Dict[str, Any], messages_key: str = 'conversation') -> Dict[str, str]:
    """Extract first human(question) and assistant(response) from conversation list."""
    msgs = rec.get(messages_key)
    res = {"question": "", "response": ""}
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get('role') or m.get('from') or m.get('speaker') or '').lower()
            content = m.get('content') or m.get('text') or ''
            if isinstance(content, list):
                try:
                    content = ' '.join(x for x in content if isinstance(x, str))
                except Exception:
                    content = str(content)
            if not res['question'] and role in ('user', 'human'):
                res['question'] = str(content)
            elif not res['response'] and role in ('assistant', 'bot', 'gpt'):
                res['response'] = str(content)
            if res['question'] and res['response']:
                break
    return res

# ---------------- Public API ----------------

def _build_ctx(extra: Dict[str, Any]) -> Dict[str, Any]:
    ctx = dict(_ALLOWED_BUILTINS)
    ctx.update(_ALLOWED_MODULES)
    ctx.update(extra or {})
    return ctx

def coalesce(*values: Any, default: Any = None) -> Any:
    """Return the first value that is not None and not an empty string, else default."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        return v
    return default

def eval_expr_safe(expr: str, rec: Dict[str, Any], idx: int = 0) -> Any:
    """Validate & evaluate expr safely with helpers in context."""
    if not isinstance(expr, str) or not expr.strip():
        return None
    tree = ast.parse(expr, mode='eval')
    SafeValidator().visit(tree)
    code = compile(tree, '<expr>', 'eval')
    ctx = _build_ctx({
        'rec': rec,
        'idx': idx,
        'get': get,
        'join_text': join_text,
        'chat_to_qa': chat_to_qa,
        'coalesce': coalesce,
        # text
        'lower': lower,
        'upper': upper,
        'strip': strip,
        'replace': replace,
        'split': split,
        'join_list': join_list,
        'contains': contains,
        'startswith': startswith,
        'endswith': endswith,
        'truncate': truncate,
        # json
        'json_loads': json_loads,
        'json_dumps': json_dumps,
        # list/collection
        'flatten': flatten,
        # numeric
        'clamp': clamp,
        'coalesce': coalesce,
        # regex
        'regex_search': regex_search,
        'regex_findall': regex_findall,
        'regex_sub': regex_sub,
        # datetime
        'date_parse': date_parse,
        'date_format': date_format,
        'to_timestamp': to_timestamp,
        'date_diff': date_diff,
        # dict lookup
        'dict_lookup': dict_lookup,
        # casting
        'to_int': to_int,
        'to_float': to_float,
        'safe_number': safe_number,
        # Excel-style text
        'LEFT': LEFT, 'RIGHT': RIGHT, 'MID': MID, 'LEN': LEN, 'TRIM': TRIM,
        'LOWER': LOWER, 'UPPER': UPPER, 'PROPER': PROPER, 'SUBSTITUTE': SUBSTITUTE,
        'REPLACE': REPLACE, 'FIND': FIND, 'SEARCH': SEARCH, 'CONCAT': CONCAT,
        'CONCATENATE': CONCATENATE, 'TEXTSPLIT': TEXTSPLIT, 'TEXTAFTER': TEXTAFTER,
        'TEXTBEFORE': TEXTBEFORE, 'VALUE': VALUE, 'TEXT': TEXT,
        # Excel-style logic
        'IF': IF, 'AND': AND, 'OR': OR, 'NOT': NOT, 'IFERROR': IFERROR, 'IFNA': IFNA,
        # Excel-style math/stat
        'SUM': SUM, 'AVERAGE': AVERAGE, 'MEDIAN': MEDIAN, 'MODE': MODE,
        'ROUNDDOWN': ROUNDDOWN, 'ROUNDUP': ROUNDUP, 'ROUND': ROUND,
        'INT': INT, 'MOD': MOD, 'POWER': POWER, 'SQRT': SQRT, 'RAND': RAND, 'RANDBETWEEN': RANDBETWEEN,
        'COUNT': COUNT, 'COUNTA': COUNTA, 'COUNTBLANK': COUNTBLANK,
        'MIN': MIN_, 'MAX': MAX_, 'SMALL': SMALL, 'LARGE': LARGE,
        'CEILING': CEILING, 'FLOOR': FLOOR,
        # Excel-style date/time
        'DATE': DATE, 'TIME': TIME, 'TODAY': TODAY, 'NOW': NOW, 'YEAR': YEAR, 'MONTH': MONTH,
        'DAY': DAY, 'HOUR': HOUR, 'MINUTE': MINUTE, 'SECOND': SECOND, 'WEEKDAY': WEEKDAY, 'EOMONTH': EOMONTH,
        # Excel-style array
        'TEXTJOIN': TEXTJOIN, 'UNIQUE': UNIQUE, 'FILTER': FILTER, 'SORT': SORT,
        'INDEX': INDEX, 'MATCH': MATCH, 'TRANSPOSE': TRANSPOSE, 'CHOOSE': CHOOSE,
        # Excel-style criteria aggregations
        'COUNTIF': COUNTIF, 'SUMIF': SUMIF, 'AVERAGEIF': AVERAGEIF,
        'COUNTIFS': COUNTIFS, 'SUMIFS': SUMIFS, 'AVERAGEIFS': AVERAGEIFS,
        # Excel-style info/text extras
        'EXACT': EXACT, 'CLEAN': CLEAN, 'REPT': REPT, 'CHAR': CHAR, 'CODE': CODE,
        'ISBLANK': ISBLANK, 'ISNUMBER': ISNUMBER, 'ISTEXT': ISTEXT, 'ISLOGICAL': ISLOGICAL, 'ISNA': ISNA, 'ISNONTEXT': ISNONTEXT,
        # HELP
        'HELP': HELP,
        # Template helpers
        'run_template': run_template,
        'run_template_by_name': run_template_by_name,
        })
    return eval(code, {"__builtins__": {}}, ctx)

# ================= Template Runner =================
def _template_exec_env() -> Dict[str, Any]:
    """Build a restricted exec environment for running template code."""
    return _build_ctx({
        # helpers identical to eval context
        'get': get,
        'join_text': join_text,
        'chat_to_qa': chat_to_qa,
        'coalesce': coalesce,
        'lower': lower, 'upper': upper, 'strip': strip, 'replace': replace, 'split': split,
        'join_list': join_list, 'contains': contains, 'startswith': startswith, 'endswith': endswith, 'truncate': truncate,
        'json_loads': json_loads, 'json_dumps': json_dumps,
        'flatten': flatten,
        'to_int': to_int, 'to_float': to_float, 'safe_number': safe_number,
        'regex_search': regex_search, 'regex_findall': regex_findall, 'regex_sub': regex_sub,
        'date_parse': date_parse, 'date_format': date_format, 'to_timestamp': to_timestamp, 'date_diff': date_diff,
        'dict_lookup': dict_lookup,
        # Excel-like
        'LEFT': LEFT, 'RIGHT': RIGHT, 'MID': MID, 'LEN': LEN, 'TRIM': TRIM, 'LOWER': LOWER, 'UPPER': UPPER,
        'PROPER': PROPER, 'SUBSTITUTE': SUBSTITUTE, 'REPLACE': REPLACE, 'FIND': FIND, 'SEARCH': SEARCH,
        'CONCAT': CONCAT, 'CONCATENATE': CONCATENATE, 'TEXTSPLIT': TEXTSPLIT, 'TEXTAFTER': TEXTAFTER, 'TEXTBEFORE': TEXTBEFORE,
        'VALUE': VALUE, 'TEXT': TEXT, 'CLEAN': CLEAN, 'EXACT': EXACT, 'REPT': REPT, 'CHAR': CHAR, 'CODE': CODE,
        'IF': IF, 'AND': AND, 'OR': OR, 'NOT': NOT, 'IFERROR': IFERROR, 'IFNA': IFNA,
        'SUM': SUM, 'AVERAGE': AVERAGE, 'MEDIAN': MEDIAN, 'MODE': MODE, 'ROUND': ROUND, 'ROUNDDOWN': ROUNDDOWN, 'ROUNDUP': ROUNDUP,
        'INT': INT, 'MOD': MOD, 'POWER': POWER, 'SQRT': SQRT, 'RAND': RAND, 'RANDBETWEEN': RANDBETWEEN,
        'COUNT': COUNT, 'COUNTA': COUNTA, 'COUNTBLANK': COUNTBLANK, 'MIN': MIN_, 'MAX': MAX_, 'SMALL': SMALL, 'LARGE': LARGE,
        'CEILING': CEILING, 'FLOOR': FLOOR,
        'TEXTJOIN': TEXTJOIN, 'UNIQUE': UNIQUE, 'FILTER': FILTER, 'SORT': SORT, 'INDEX': INDEX, 'MATCH': MATCH,
        'TRANSPOSE': TRANSPOSE, 'CHOOSE': CHOOSE,
        'COUNTIF': COUNTIF, 'SUMIF': SUMIF, 'AVERAGEIF': AVERAGEIF, 'COUNTIFS': COUNTIFS, 'SUMIFS': SUMIFS, 'AVERAGEIFS': AVERAGEIFS,
        'HELP': HELP,
    })

def _call_template_fn(function_code: str, rec: Dict[str, Any]) -> Any:
    """Exec template code and call first available function among transform/main/apply."""
    env = _template_exec_env()
    ns: Dict[str, Any] = {}
    exec(function_code or "", {"__builtins__": {}}, ns)  # restricted
    for fn_name in ("transform", "main", "apply"):
        fn = ns.get(fn_name)
        if callable(fn):
            return fn(rec)
    raise ValueError("No callable function found in template. Define 'transform', 'main' or 'apply'.")

def run_template(template_id: str, rec: Dict[str, Any]) -> Any:
    """Run a saved function template by ID with the given record.

    Example usage in函数框: run_template("<uuid>", rec)
    """
    try:
        from tools.dataset.func_templates import FunctionTemplateManager  # local import
        tm = FunctionTemplateManager()
        tpl = tm.get_template(str(template_id))
        if not tpl:
            raise ValueError("Template not found: " + str(template_id))
        return _call_template_fn(tpl.get("function_code", ""), rec)
    except Exception as e:
        # Return error text to preview而不是抛出
        return f"<TEMPLATE ERROR: {e}>"

def run_template_by_name(name: str, rec: Dict[str, Any]) -> Any:
    """Run the most recently更新的模板 by name (case-insensitive)."""
    try:
        from tools.dataset.func_templates import FunctionTemplateManager
        tm = FunctionTemplateManager()
        name_lc = str(name).strip().lower()
        candidates = [t for t in tm.list_templates() if str(t.get("name", "")).strip().lower() == name_lc]
        if not candidates:
            raise ValueError("Template not found by name: " + str(name))
        # 已按更新时间降序排序过 list_templates；取第一个
        tpl = candidates[0]
        return _call_template_fn(tpl.get("function_code", ""), rec)
    except Exception as e:
        return f"<TEMPLATE ERROR: {e}>"

__all__ = [
    'eval_expr_safe',
    'run_template', 'run_template_by_name',
    # record helpers
    'get', 'join_text', 'chat_to_qa', 'coalesce',
    # text
    'lower', 'upper', 'strip', 'replace', 'split', 'join_list', 'contains', 'startswith', 'endswith', 'truncate',
    # json
    'json_loads', 'json_dumps',
    # list/collection
    'flatten',
    # numeric
    'clamp',
    # regex
    'regex_search', 'regex_findall', 'regex_sub',
    # datetime
    'date_parse', 'date_format', 'to_timestamp', 'date_diff',
    # dict
    'dict_lookup',
    # casting
    'to_int', 'to_float', 'safe_number',
    # Excel-like text
    'LEFT', 'RIGHT', 'MID', 'LEN', 'TRIM', 'LOWER', 'UPPER', 'PROPER', 'SUBSTITUTE', 'REPLACE',
    'FIND', 'SEARCH', 'CONCAT', 'CONCATENATE', 'TEXTSPLIT', 'TEXTAFTER', 'TEXTBEFORE', 'VALUE', 'TEXT',
    # Excel-like logic
    'IF', 'AND', 'OR', 'NOT', 'IFERROR', 'IFNA',
    # Excel-like math/stat
    'SUM', 'AVERAGE', 'MEDIAN', 'MODE', 'ROUNDDOWN', 'ROUNDUP', 'ROUND', 'INT', 'MOD', 'POWER', 'SQRT', 'RAND', 'RANDBETWEEN',
    'COUNT', 'COUNTA', 'COUNTBLANK', 'MIN', 'MAX', 'SMALL', 'LARGE', 'CEILING', 'FLOOR',
    # Excel-like date/time
    'DATE', 'TIME', 'TODAY', 'NOW', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'WEEKDAY', 'EOMONTH',
    # Excel-like array
    'TEXTJOIN', 'UNIQUE', 'FILTER', 'SORT', 'INDEX', 'MATCH', 'TRANSPOSE', 'CHOOSE',
    # Excel-like criteria aggregations
    'COUNTIF', 'SUMIF', 'AVERAGEIF', 'COUNTIFS', 'SUMIFS', 'AVERAGEIFS',
    # Excel-like info/text extras
    'EXACT', 'CLEAN', 'REPT', 'CHAR', 'CODE', 'ISBLANK', 'ISNUMBER', 'ISTEXT', 'ISLOGICAL', 'ISNA', 'ISNONTEXT',
    # HELP
    'HELP',
]
