"""The schema catalogue: models, loader and validation against the live database.

schema/catalogue.yaml is the single description of the database that the prompt
(context_builder) and the V1 schema check are built from. An entry is either one
column or a numbered family written as "PREFIX_1..10", which stands for
PREFIX_1, PREFIX_2, ... PREFIX_10.

Usage (prints a validation report; exit code 1 if catalogue and DB disagree):
    python -m claimssql.schema.catalogue
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from claimssql.config import REPO_ROOT

if TYPE_CHECKING:  # loading a catalogue (e.g. for generation) must not need duckdb
    import duckdb

DEFAULT_CATALOGUE_PATH = REPO_ROOT / "schema" / "catalogue.yaml"

_FAMILY = re.compile(r"^(?P<prefix>.+_)(?P<first>\d+)\.\.(?P<last>\d+)$")


def family_name(prefix: str, first: int, last: int) -> str:
    return f"{prefix}{first}..{last}"


class ColumnEntry(BaseModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    description: str = ""
    include: bool = True
    codes: dict[str, str] = Field(default_factory=dict)
    notes: str = ""

    @field_validator("codes", mode="before")
    @classmethod
    def _codes_as_text(cls, value: object) -> object:
        # YAML reads an unquoted `1: Male` as an integer key; codes are text.
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        return value or {}

    @field_validator("description", "notes", mode="before")
    @classmethod
    def _none_is_empty(cls, value: object) -> object:
        return "" if value is None else value

    @model_validator(mode="after")
    def _family_range_valid(self) -> ColumnEntry:
        match = _FAMILY.match(self.name)
        if match and int(match["first"]) > int(match["last"]):
            raise ValueError(f"family {self.name!r}: first number is above the last")
        return self

    @property
    def is_family(self) -> bool:
        return _FAMILY.match(self.name) is not None

    def members(self) -> list[str]:
        """Real column names: the family expanded, or just the name."""
        match = _FAMILY.match(self.name)
        if not match:
            return [self.name]
        return [
            f"{match['prefix']}{n}" for n in range(int(match["first"]), int(match["last"]) + 1)
        ]


class TableEntry(BaseModel):
    name: str = Field(min_length=1)
    grain: str = ""
    description: str = ""
    columns: list[ColumnEntry] = Field(min_length=1)

    def column_types(self, included_only: bool = False) -> dict[str, str]:
        """Every real column name -> type."""
        return {
            member: entry.type
            for entry in self.columns
            if entry.include or not included_only
            for member in entry.members()
        }


class Catalogue(BaseModel):
    source: dict[str, str] = Field(default_factory=dict)
    tables: list[TableEntry] = Field(min_length=1)

    def table(self, name: str) -> TableEntry:
        for table in self.tables:
            if table.name.lower() == name.lower():
                return table
        raise KeyError(f"no table {name!r} in the catalogue")

    def n_entries(self) -> int:
        return sum(len(t.columns) for t in self.tables)


def load_catalogue(path: Path | None = None) -> Catalogue:
    path = path or DEFAULT_CATALOGUE_PATH
    with path.open(encoding="utf-8") as f:
        return Catalogue.model_validate(yaml.safe_load(f))


# ------------------------------------------------------------ validation ----


@dataclass
class ValidationReport:
    # Mismatches: catalogue and database disagree, must be fixed.
    tables_not_in_catalogue: list[str] = field(default_factory=list)
    tables_not_in_db: list[str] = field(default_factory=list)
    columns_not_in_catalogue: list[tuple[str, str]] = field(default_factory=list)
    columns_not_in_db: list[tuple[str, str]] = field(default_factory=list)
    duplicate_columns: list[tuple[str, str]] = field(default_factory=list)
    type_mismatches: list[tuple[str, str, str, str]] = field(default_factory=list)
    # Warnings: for the owner to review, not errors.
    empty_descriptions: list[tuple[str, str]] = field(default_factory=list)
    values_without_code: list[tuple[str, str, list[str]]] = field(default_factory=list)
    codes_not_in_data: list[tuple[str, str, list[str]]] = field(default_factory=list)

    _MISMATCH_FIELDS = (
        "tables_not_in_catalogue", "tables_not_in_db", "columns_not_in_catalogue",
        "columns_not_in_db", "duplicate_columns", "type_mismatches",
    )  # fmt: skip
    _WARNING_FIELDS = ("empty_descriptions", "values_without_code", "codes_not_in_data")

    @property
    def has_mismatches(self) -> bool:
        return any(getattr(self, name) for name in self._MISMATCH_FIELDS)

    def lines(self) -> list[str]:
        out = []
        for kind, names in (("MISMATCH", self._MISMATCH_FIELDS), ("warning", self._WARNING_FIELDS)):
            for name in names:
                for item in getattr(self, name):
                    detail = item if isinstance(item, str) else " | ".join(map(str, item))
                    out.append(f"{kind:<8} {name}: {detail}")
        return out


def _db_tables(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, str]]:
    """table -> {column: type} for every table in the database."""
    rows = con.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
    ).fetchall()
    tables: dict[str, dict[str, str]] = {}
    for table, column, data_type in rows:
        tables.setdefault(table, {})[column] = data_type
    return tables


def _distinct_text_values(con: duckdb.DuckDBPyConnection, table: str, column: str) -> set[str]:
    rows = con.execute(
        f'SELECT DISTINCT CAST("{column}" AS VARCHAR) FROM "{table}" WHERE "{column}" IS NOT NULL'
    ).fetchall()
    return {r[0] for r in rows}


def validate_against_db(
    catalogue: Catalogue, con: duckdb.DuckDBPyConnection
) -> ValidationReport:
    report = ValidationReport()
    db = _db_tables(con)
    db_lower = {name.lower(): name for name in db}
    cat_lower = {t.name.lower() for t in catalogue.tables}

    report.tables_not_in_catalogue = [n for n in db if n.lower() not in cat_lower]

    for table in catalogue.tables:
        if not table.description.strip():
            report.empty_descriptions.append((table.name, "(table description)"))
        if not table.grain.strip():
            report.empty_descriptions.append((table.name, "(table grain)"))
        if table.name.lower() not in db_lower:
            report.tables_not_in_db.append(table.name)
            continue

        db_table = db_lower[table.name.lower()]
        # SQL identifiers are case-insensitive, so compare upper-cased names.
        db_columns = {c.upper(): (c, t) for c, t in db[db_table].items()}
        seen: set[str] = set()
        for entry in table.columns:
            if not entry.description.strip():
                report.empty_descriptions.append((table.name, entry.name))
            present = []
            for member in entry.members():
                key = member.upper()
                if key in seen:
                    report.duplicate_columns.append((table.name, member))
                seen.add(key)
                if key not in db_columns:
                    report.columns_not_in_db.append((table.name, member))
                    continue
                real_name, db_type = db_columns[key]
                present.append(real_name)
                if db_type.upper() != entry.type.upper():
                    report.type_mismatches.append((table.name, member, entry.type, db_type))
            if entry.codes and present:
                values: set[str] = set()
                for column in present:
                    values |= _distinct_text_values(con, db_table, column)
                if missing := sorted(values - set(entry.codes)):
                    report.values_without_code.append((table.name, entry.name, missing))
                if unused := sorted(set(entry.codes) - values):
                    report.codes_not_in_data.append((table.name, entry.name, unused))
        report.columns_not_in_catalogue += [
            (table.name, real) for key, (real, _) in db_columns.items() if key not in seen
        ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the catalogue against the database.")
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE_PATH)
    parser.add_argument("--db-path", type=Path, help="default: data/claims.duckdb")
    args = parser.parse_args()

    from claimssql.db.connection import DEFAULT_DB_PATH, get_connection

    args.db_path = args.db_path or DEFAULT_DB_PATH
    catalogue = load_catalogue(args.catalogue)
    con = get_connection(read_only=True, db_path=args.db_path)
    try:
        report = validate_against_db(catalogue, con)
    finally:
        con.close()

    for line in report.lines():
        print(line)
    included = sum(e.include for t in catalogue.tables for e in t.columns)
    print(f"\n{len(catalogue.tables)} tables, {catalogue.n_entries()} entries "
          f"({included} included) checked against {args.db_path}")
    print("Result: " + ("MISMATCHES found" if report.has_mismatches else "catalogue matches DB"))
    return 1 if report.has_mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
