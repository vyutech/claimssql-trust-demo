"""Build data/demo.duckdb: a small copy of the claims database for the live demo.

It holds a fixed random set of beneficiaries with ALL of their rows: every
year of the beneficiary table and every inpatient, outpatient and carrier
claim. Tables, columns and types are exactly those of the full database, so
the same catalogue (and the same verifiers) apply.

"Fixed random": beneficiaries are ordered by md5(id + salt) and the first n are
taken. md5 is the same on every machine and DuckDB version, so the demo
database always holds the same people.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

from claimssql.db.load_synpuf import TABLE_SOURCES

DEFAULT_N = 5000
DEFAULT_SALT = "claimssql-demo-v1"


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_demo_db(
    source: Path, out: Path, n: int = DEFAULT_N, salt: str = DEFAULT_SALT
) -> dict[str, int]:
    """Write `out` from `source`; returns row counts per table. `out` is replaced."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    for stale in (tmp, Path(str(tmp) + ".wal")):
        stale.unlink(missing_ok=True)

    con = duckdb.connect(str(tmp))
    try:
        con.execute(f"ATTACH {_sql_str(source.as_posix())} AS src (READ_ONLY)")
        con.execute(f"""
            CREATE TEMP TABLE chosen AS
            SELECT DESYNPUF_ID FROM (SELECT DISTINCT DESYNPUF_ID FROM src.beneficiary)
            ORDER BY md5(DESYNPUF_ID || {_sql_str(salt)}) LIMIT {int(n)}
        """)
        counts = {}
        for table in TABLE_SOURCES:
            # ORDER BY keeps the file's contents in a stable order between builds.
            con.execute(f"""
                CREATE TABLE {table} AS
                SELECT * FROM src.{table}
                WHERE DESYNPUF_ID IN (SELECT DESYNPUF_ID FROM chosen)
                ORDER BY ALL
            """)
            counts[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        con.execute("DETACH src")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    os.replace(tmp, out)
    return counts


def chosen_ids(con: duckdb.DuckDBPyConnection) -> list[str]:
    """The beneficiaries in a demo database, sorted."""
    rows = con.execute("SELECT DISTINCT DESYNPUF_ID FROM beneficiary ORDER BY 1").fetchall()
    return [r[0] for r in rows]
