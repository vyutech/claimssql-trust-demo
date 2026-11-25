"""Profile every column of the claims database.

Writes two files to results/tables/:
  * column_profile.csv - one row per column: type, null %, distinct count,
    min, max and the 5 most frequent values.
  * low_cardinality_values.json - {table: {column: [values]}} for columns with
    at most --max-distinct distinct values. The hallucination taxonomy (P6)
    uses it to spot literals a column never takes.

Usage:
    python -m claimssql.db.profile
    python -m claimssql.db.profile --max-distinct 30 --tables beneficiary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import astuple, dataclass
from pathlib import Path

import duckdb

from claimssql.config import RESULTS_TABLES_DIR
from claimssql.db.connection import DEFAULT_DB_PATH, get_connection
from claimssql.db.load_synpuf import existing_tables, write_csv

N_SAMPLES = 5
SAMPLE_SEPARATOR = " | "
PROFILE_HEADER = [
    "table", "column", "type", "null_pct", "distinct_count", "min", "max", "sample_values",
]  # fmt: skip


@dataclass(frozen=True)
class ColumnProfile:
    table: str
    column: str
    type: str
    null_pct: float
    distinct_count: int
    min: str | None
    max: str | None
    sample_values: str


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def profile_column(
    con: duckdb.DuckDBPyConnection, table: str, column: str, col_type: str, total_rows: int
) -> ColumnProfile:
    col = _q(column)
    # count(col) skips NULLs; distinct count is exact because the taxonomy's
    # low-cardinality cut-off depends on it. min/max of VARCHAR are alphabetical.
    non_null, distinct, lo, hi = con.execute(
        f"SELECT count({col}), count(DISTINCT {col}), "
        f"CAST(min({col}) AS VARCHAR), CAST(max({col}) AS VARCHAR) FROM {table}"
    ).fetchone()
    # Most frequent values, ties broken by value, so re-runs give the same file.
    samples = con.execute(
        f"SELECT CAST({col} AS VARCHAR) FROM {table} WHERE {col} IS NOT NULL "
        f"GROUP BY {col} ORDER BY count(*) DESC, {col} LIMIT {N_SAMPLES}"
    ).fetchall()
    null_pct = round(100 * (total_rows - non_null) / total_rows, 4) if total_rows else 0.0
    return ColumnProfile(
        table=table,
        column=column,
        type=col_type,
        null_pct=null_pct,
        distinct_count=distinct,
        min=lo,
        max=hi,
        sample_values=SAMPLE_SEPARATOR.join(r[0] for r in samples),
    )


def profile_table(con: duckdb.DuckDBPyConnection, table: str) -> list[ColumnProfile]:
    total = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    columns = [(r[0], r[1]) for r in con.execute(f"DESCRIBE {table}").fetchall()]
    return [profile_column(con, table, name, typ, total) for name, typ in columns]


def distinct_values(con: duckdb.DuckDBPyConnection, table: str, column: str) -> list:
    """Sorted non-NULL distinct values in their native type (str / int / float)."""
    col = _q(column)
    rows = con.execute(
        f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY 1"
    ).fetchall()
    return [r[0] for r in rows]


def low_cardinality_values(
    con: duckdb.DuckDBPyConnection, profiles: list[ColumnProfile], max_distinct: int
) -> dict[str, dict[str, list]]:
    out: dict[str, dict[str, list]] = {}
    for p in profiles:
        if p.distinct_count <= max_distinct:
            out.setdefault(p.table, {})[p.column] = distinct_values(con, p.table, p.column)
    return out


def write_profile(path: Path, profiles: list[ColumnProfile]) -> None:
    write_csv(path, PROFILE_HEADER, [astuple(p) for p in profiles])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_TABLES_DIR)
    parser.add_argument("--max-distinct", type=int, default=30,
                        help="columns with at most this many distinct values go in the JSON")
    parser.add_argument("--tables", nargs="+", help="profile only these tables")
    args = parser.parse_args()

    con = get_connection(read_only=True, db_path=args.db_path)
    try:
        tables = existing_tables(con)
        if args.tables:
            unknown = sorted(set(args.tables) - set(tables))
            if unknown:
                parser.error(f"not in the database: {', '.join(unknown)}")
            tables = [t for t in tables if t in args.tables]

        profiles: list[ColumnProfile] = []
        for table in tables:
            start = time.perf_counter()
            table_profiles = profile_table(con, table)
            profiles.extend(table_profiles)
            print(f"{table}: {len(table_profiles)} columns in "
                  f"{time.perf_counter() - start:.1f} s")
        low_card = low_cardinality_values(con, profiles, args.max_distinct)
    finally:
        con.close()

    write_profile(args.out_dir / "column_profile.csv", profiles)
    json_path = args.out_dir / "low_cardinality_values.json"
    # default=str covers any value type json cannot write natively (e.g. Decimal).
    json_path.write_text(json.dumps(low_card, indent=1, default=str), encoding="utf-8")

    n_low = sum(len(cols) for cols in low_card.values())
    print(f"\nWrote {args.out_dir / 'column_profile.csv'} ({len(profiles)} columns)")
    print(f"Wrote {json_path} ({n_low} columns with <= {args.max_distinct} distinct values)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
