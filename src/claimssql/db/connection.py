"""One place that opens the claims database.

Everything that only reads (profiling, verifiers, scoring, the API) asks for a
read-only connection, so model-generated SQL can never modify the data.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from claimssql.config import REPO_ROOT

DEFAULT_DB_PATH = REPO_ROOT / "data" / "claims.duckdb"


def get_connection(
    read_only: bool = True, db_path: Path | str | None = None
) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB file. Read-only by default; only the loader writes."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if read_only and not path.exists():
        # DuckDB's own error for this is cryptic; say what to do instead.
        raise FileNotFoundError(
            f"{path} not found - build it with: python -m claimssql.db.load_synpuf"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)
