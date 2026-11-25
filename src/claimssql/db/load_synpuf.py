"""Build data/claims.duckdb from the CMS DE-SynPUF Sample 1 CSV files.

DuckDB reads the CSVs directly (no pandas). Column names are kept exactly as in
the files. Types are forced by column NAME for three groups, because DuckDB's
auto-detection would otherwise turn codes into numbers and drop leading zeros:

  * IDs and codes  -> VARCHAR   (CLM_ID, *_NPI, *_CD, SP_STATE_CODE, ...)
  * *_DT dates     -> BIGINT    (raw YYYYMMDD; converting them is the query's job)
  * PLAN_CVRG_MOS_NUM -> INTEGER (a months count stored as '00'..'12' in the file)

Everything else (amounts, counts, flags) keeps the auto-detected type.

Usage:
    python -m claimssql.db.load_synpuf
    python -m claimssql.db.load_synpuf --tables beneficiary inpatient_claims
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from claimssql.config import REPO_ROOT, RESULTS_TABLES_DIR
from claimssql.db.connection import DEFAULT_DB_PATH, get_connection

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"

# table -> source files; several files are stacked with UNION ALL.
TABLE_SOURCES: dict[str, list[str]] = {
    "beneficiary": [
        "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv",
        "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv",
        "DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv",
    ],
    "inpatient_claims": ["DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"],
    "outpatient_claims": ["DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv"],
    "carrier_claims": [
        "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv",
        "DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv",
    ],
}

# Tables whose files are one-per-year and need a `year` column from the file name.
YEARLY_TABLES = {"beneficiary"}
_YEAR_IN_NAME = re.compile(r"DE1_0_(\d{4})_Beneficiary")

# VARCHAR by exact name. BENE_ESRD_IND is not a code column by name, but its
# values are '0'/'Y', so it has to be text.
_VARCHAR_NAMES = {
    "DESYNPUF_ID",
    "CLM_ID",
    "PRVDR_NUM",
    "SP_STATE_CODE",
    "BENE_COUNTY_CD",
    "BENE_ESRD_IND",
}
# VARCHAR by pattern: any *_CD or *_NPI column, with or without a numeric
# suffix (ICD9_DGNS_CD_3, PRF_PHYSN_NPI_12), plus the TAX_NUM_n identifiers.
_VARCHAR_PATTERN = re.compile(r"(_CD|_NPI)(_\d+)?$|^TAX_NUM_\d+$")
_DATE_PATTERN = re.compile(r"_DT$")
_INTEGER_NAMES = {"PLAN_CVRG_MOS_NUM"}


@dataclass(frozen=True)
class LoadResult:
    table: str
    seconds: float
    varchar_columns: list[str]
    files: list[str]


def column_type(name: str) -> str | None:
    """Forced DuckDB type for a column name, or None to keep auto-detection."""
    upper = name.upper()
    if upper in _VARCHAR_NAMES or _VARCHAR_PATTERN.search(upper):
        return "VARCHAR"
    if _DATE_PATTERN.search(upper):
        return "BIGINT"
    if upper in _INTEGER_NAMES:
        return "INTEGER"
    return None


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _read_csv_sql(path: Path, columns: list[str]) -> str:
    forced = {c: t for c in columns if (t := column_type(c))}
    types = ", ".join(f"{_sql_str(c)}: {_sql_str(t)}" for c, t in forced.items())
    # as_posix(): forward slashes work on Windows and Linux alike.
    return f"read_csv({_sql_str(path.as_posix())}, header = true, types = {{{types}}})"


def build_table_sql(table: str, paths: list[Path], columns: list[str]) -> str:
    """CREATE OR REPLACE TABLE ... AS <one SELECT per file, UNION ALL>."""
    selects = []
    for path in paths:
        extra = ""
        if table in YEARLY_TABLES:
            match = _YEAR_IN_NAME.search(path.name)
            if not match:
                raise ValueError(f"cannot read the year from file name {path.name}")
            extra = f", CAST({int(match.group(1))} AS INTEGER) AS year"
        selects.append(f"SELECT *{extra} FROM {_read_csv_sql(path, columns)}")
    body = "\nUNION ALL\n".join(selects)
    return f"CREATE OR REPLACE TABLE {table} AS\n{body}"


def load_table(con: duckdb.DuckDBPyConnection, table: str, raw_dir: Path) -> LoadResult:
    paths = [raw_dir / name for name in TABLE_SOURCES[table]]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{table}: missing in {raw_dir}: {', '.join(missing)}")

    columns = read_header(paths[0])
    for p in paths[1:]:
        # UNION ALL matches by position, so differing headers would silently
        # put values in the wrong column. Refuse instead.
        if read_header(p) != columns:
            raise ValueError(f"{table}: header of {p.name} differs from {paths[0].name}")

    start = time.perf_counter()
    con.execute(build_table_sql(table, paths, columns))
    return LoadResult(
        table=table,
        seconds=time.perf_counter() - start,
        varchar_columns=[c for c in columns if column_type(c) == "VARCHAR"],
        files=[p.name for p in paths],
    )


def build_database(
    raw_dir: Path, db_path: Path, tables: list[str] | None = None
) -> list[LoadResult]:
    con = get_connection(read_only=False, db_path=db_path)
    try:
        return [load_table(con, t, raw_dir) for t in (tables or list(TABLE_SOURCES))]
    finally:
        con.close()


# ---------------------------------------------------------------- checks ----


def existing_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    present = {r[0] for r in rows}
    return [t for t in TABLE_SOURCES if t in present]


def row_counts(con: duckdb.DuckDBPyConnection) -> list[tuple[str, int, int]]:
    """(table, row_count, column_count) for every loaded table."""
    out = []
    for table in existing_tables(con):
        rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        cols = len(con.execute(f"DESCRIBE {table}").fetchall())
        out.append((table, rows, cols))
    return out


def beneficiary_ids_per_year(con: duckdb.DuckDBPyConnection) -> list[tuple[int, int, int]]:
    """(year, rows, distinct DESYNPUF_ID). Equal numbers = one row per person per year."""
    return con.execute(
        "SELECT year, count(*), count(DISTINCT DESYNPUF_ID) "
        "FROM beneficiary GROUP BY year ORDER BY year"
    ).fetchall()


def multi_row_claim_ids(con: duckdb.DuckDBPyConnection, table: str) -> tuple[int, int, int]:
    """(distinct CLM_IDs, CLM_IDs with more than one row, max rows for one CLM_ID)."""
    return con.execute(
        f"""
        SELECT count(*), count(*) FILTER (WHERE n > 1), coalesce(max(n), 0)
        FROM (SELECT CLM_ID, count(*) AS n FROM {table} GROUP BY CLM_ID)
        """
    ).fetchone()


def run_checks(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str, int]]:
    """Long-format rows (check, table, key, value) for results/tables/load_checks.csv."""
    tables = existing_tables(con)
    out: list[tuple[str, str, str, int]] = []
    if "beneficiary" in tables:
        for year, rows, ids in beneficiary_ids_per_year(con):
            out.append(("rows", "beneficiary", str(year), rows))
            out.append(("distinct_desynpuf_id", "beneficiary", str(year), ids))
    for table in ("inpatient_claims", "outpatient_claims"):
        if table in tables:
            distinct, multi, max_rows = multi_row_claim_ids(con, table)
            out.append(("distinct_clm_id", table, "", distinct))
            out.append(("clm_ids_with_multiple_rows", table, "", multi))
            out.append(("max_rows_per_clm_id", table, "", max_rows))
    return out


def count_csv_rows(path: Path) -> int:
    """Data rows in a CSV by counting newlines - an independent check on the load."""
    newlines = 0
    last = b"\n"
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            newlines += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":  # final line without a trailing newline
        newlines += 1
    return newlines - 1  # minus the header


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_TABLES_DIR)
    parser.add_argument(
        "--tables", nargs="+", choices=list(TABLE_SOURCES), help="load only these tables"
    )
    parser.add_argument(
        "--skip-line-check", action="store_true", help="skip comparing with CSV line counts"
    )
    args = parser.parse_args()

    for result in build_database(args.raw_dir, args.db_path, args.tables):
        print(f"{result.table}: loaded {len(result.files)} file(s) in {result.seconds:.1f} s")
        print(f"  VARCHAR forced on {len(result.varchar_columns)} columns: "
              f"{', '.join(result.varchar_columns)}")

    con = get_connection(read_only=True, db_path=args.db_path)
    try:
        counts = row_counts(con)
        checks = run_checks(con)
    finally:
        con.close()

    ok = True
    print("\nRow counts")
    for table, rows, cols in counts:
        line = f"  {table:<18} {rows:>10,} rows  {cols:>3} columns"
        if not args.skip_line_check:
            expected = sum(count_csv_rows(args.raw_dir / n) for n in TABLE_SOURCES[table])
            match = expected == rows
            ok &= match
            line += "  (matches CSV line count)" if match else f"  MISMATCH: CSV has {expected:,}"
        print(line)

    print("\nChecks")
    for check, table, key, value in checks:
        print(f"  {table:<18} {check:<28} {key:<5} {value:>10,}")

    write_csv(args.out_dir / "row_counts.csv", ["table", "row_count", "column_count"], counts)
    write_csv(args.out_dir / "load_checks.csv", ["check", "table", "key", "value"], checks)
    size_mb = args.db_path.stat().st_size / 1e6
    print(f"\nWrote {args.out_dir / 'row_counts.csv'} and load_checks.csv")
    print(f"Database: {args.db_path} ({size_mb:,.1f} MB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
