"""Run SQL and compare result sets. Used for gold SQL, V3 and scoring alike, so
"same answer" means the same thing everywhere.

Two results are equal when their normalised forms are equal:
  * column names are ignored (aliases do not matter); column ORDER is kept
  * rows are compared as a multiset: order ignored, duplicates count
  * every number is a float rounded to 4 decimal places, so 5, 5.0 and
    Decimal('5.00') are the same value (owner decision, P3)
  * strings are stripped; NULL becomes a fixed sentinel
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dtime
from decimal import Decimal
from typing import Any

import duckdb

FLOAT_DP = 4
MAX_ROWS = 10_000
# Sentinels are strings no real cell is expected to contain.
NULL_SENTINEL = "␀NULL"
NAN_SENTINEL = "␀NaN"
_EXACT_INT_LIMIT = 2**53  # above this a float can no longer hold an integer exactly


def normalise_value(value: Any) -> Any:
    if value is None:
        return NULL_SENTINEL
    if isinstance(value, bool):  # before the number check: bool is a subclass of int
        return value
    if isinstance(value, int) and abs(value) > _EXACT_INT_LIMIT:
        return value
    if isinstance(value, (int, float, Decimal)):
        number = float(value)
        if math.isnan(number):
            return NAN_SENTINEL
        if math.isinf(number):
            return "␀Infinity" if number > 0 else "␀-Infinity"
        rounded = round(number, FLOAT_DP)
        return 0.0 if rounded == 0 else rounded  # folds -0.0 into 0.0
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (datetime, date, dtime)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return [normalise_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalise_value(v) for k, v in value.items()}
    return str(value)


def _row_key(row: list) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def normalise_result(rows: list) -> list[list]:
    """Canonical form of a result set: normalised cells, rows in a fixed order."""
    normalised = [[normalise_value(cell) for cell in row] for row in rows]
    # Sort by each row's JSON text so mixed types in a column cannot break sorting.
    return sorted(normalised, key=_row_key)


def result_hash(rows: list) -> str:
    """sha256 of the canonical JSON of the normalised result."""
    canonical = json.dumps(
        normalise_result(rows), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionResult:
    rows: list[tuple]
    columns: list[str]
    error: str | None
    truncated: bool
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return self.error is None


def execute_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    timeout_s: float = 30.0,
    max_rows: int = MAX_ROWS,
) -> ExecutionResult:
    """Run one statement. Never raises: failures come back in `error`.

    At most `max_rows` rows are returned; `truncated` says there were more.
    A truncated result has no reliable hash (which rows survive is arbitrary).
    The caller chooses the connection - pass a read-only one for untrusted SQL.
    """
    timed_out = threading.Event()

    def _interrupt() -> None:
        timed_out.set()
        con.interrupt()  # makes the running query raise in the main thread

    timer = threading.Timer(timeout_s, _interrupt)
    timer.daemon = True
    rows: list[tuple] = []
    columns: list[str] = []
    error: str | None = None
    start = time.perf_counter()
    timer.start()
    try:
        con.execute(sql)
        if con.description:
            columns = [d[0] for d in con.description]
            rows = con.fetchmany(max_rows + 1)  # one extra row reveals truncation
    except Exception as exc:  # noqa: BLE001 - any failure is a result, not a crash
        if timed_out.is_set():
            error = f"timeout: query exceeded {timeout_s:g} s"
        else:
            error = f"{type(exc).__name__}: {exc}"
        rows, columns = [], []
    finally:
        timer.cancel()
    elapsed_ms = (time.perf_counter() - start) * 1000

    truncated = len(rows) > max_rows
    return ExecutionResult(
        rows=rows[:max_rows],
        columns=columns,
        error=error,
        truncated=truncated,
        elapsed_ms=elapsed_ms,
    )
