"""V2 - dry run: does the SQL actually execute?

Two safeguards before model-written SQL touches the database:
  1. a statement gate - exactly one statement, and it must be a query
     (SELECT, WITH ... SELECT, or UNION / INTERSECT / EXCEPT); anything else is
     rejected without being run;
  2. the caller passes a read-only connection, so even a query that slipped
     through could not change data.
The query then runs with a timeout (compare.execute_sql), and any error is
sorted into a small set of classes for the report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import duckdb
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from claimssql.eval.compare import MAX_ROWS, execute_sql, result_hash

DIALECT = "duckdb"

# Error classes. The first six are from the project spec; "forbidden" is for
# statements the gate refuses to run (owner decision, P5).
PARSE, BINDER, CATALOG, CONVERSION, TIMEOUT, OTHER, FORBIDDEN = (
    "parse", "binder", "catalog", "conversion", "timeout", "other", "forbidden",
)  # fmt: skip

# Statement kinds that must never appear, even nested inside a query.
_WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.Command, exp.Copy, exp.Set, exp.Pragma, exp.Attach, exp.Detach, exp.Use,
    exp.TruncateTable, exp.Transaction, exp.Commit, exp.Rollback,
)  # fmt: skip


@dataclass(frozen=True)
class DryRunResult:
    ok: bool
    error_class: str | None = None
    error: str | None = None
    row_count: int | None = None
    truncated: bool = False
    result_hash: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def statement_gate(sql: str) -> tuple[str, str] | None:
    """None if the SQL is one read-only query; otherwise (error_class, message)."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except SqlglotError as exc:
        return PARSE, str(exc).splitlines()[0]
    if not statements:
        return PARSE, "empty SQL"
    if len(statements) > 1:
        return FORBIDDEN, f"{len(statements)} statements; only a single query is allowed"
    tree = statements[0]
    if isinstance(tree, exp.Subquery):
        tree = tree.unnest()  # a query wrapped in parentheses: "(SELECT 1)"
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return FORBIDDEN, f"{type(tree).__name__.upper()} statement; only queries are allowed"
    for node in tree.walk():
        if isinstance(node, _WRITE_NODES):
            return FORBIDDEN, f"{type(node).__name__.upper()} inside the query is not allowed"
    return None


def classify_error(message: str) -> str:
    """Error class from the text compare.execute_sql returns ("<Type>: <detail>")."""
    kind = message.split(":", 1)[0].strip().lower()
    if kind.startswith("timeout"):
        return TIMEOUT
    if "parser" in kind or "syntax" in kind:
        return PARSE
    if "binder" in kind:
        return BINDER
    if "catalog" in kind:
        return CATALOG
    if "conversion" in kind:
        return CONVERSION
    return OTHER


def dry_run(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    timeout_s: float = 30.0,
    max_rows: int = MAX_ROWS,
) -> DryRunResult:
    """Gate, then execute. `con` must be a read-only connection."""
    if (rejected := statement_gate(sql)) is not None:
        error_class, message = rejected
        return DryRunResult(ok=False, error_class=error_class, error=message)

    result = execute_sql(con, sql, timeout_s=timeout_s, max_rows=max_rows)
    if not result.ok:
        return DryRunResult(
            ok=False,
            error_class=classify_error(result.error),
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )
    return DryRunResult(
        ok=True,
        row_count=len(result.rows),
        truncated=result.truncated,
        # A truncated result has no reliable hash (which rows survive is arbitrary).
        result_hash=None if result.truncated else result_hash(result.rows),
        elapsed_ms=result.elapsed_ms,
    )
