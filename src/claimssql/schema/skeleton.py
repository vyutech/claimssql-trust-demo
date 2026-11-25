"""Draft schema/catalogue.yaml from the live database, the column profile and the
CMS codebook. The owner then reviews the draft and fills in `notes`.

Where each field comes from:
  name, type     - the live database (DESCRIBE)
  description    - the codebook's label for the variable, verbatim
  codes          - the codebook's code table, limited to values present in the data
  include        - false for columns that are 100% NULL (from the profile) and for
                   the columns listed in FORCED_EXCLUDES
  notes          - always "" here; written by the owner
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import yaml

from claimssql.db.load_synpuf import existing_tables
from claimssql.schema.catalogue import Catalogue, ColumnEntry, TableEntry, family_name
from claimssql.schema.codebook import CodebookEntry

CODEBOOK_TITLE = "CMS Linkable 2008-2010 Medicare DE-SynPUF Codebook"
CODEBOOK_URL = "https://www.cms.gov/files/document/de-10-codebook.pdf-0"

# Grain as fixed in the project spec; description from the codebook's
# "Summary of Variables" sentence for each file. The clause about stacked years
# describes how load_synpuf.py builds the table.
TABLE_INFO: dict[str, tuple[str, str]] = {
    "beneficiary": (
        "one row per beneficiary per year",
        "CMS Beneficiary Summary DE-SynPUF: each record pertains to a synthetic Medicare "
        "beneficiary. The 2008, 2009 and 2010 yearly files are stacked; the year column "
        "says which file a row came from.",
    ),
    "inpatient_claims": (
        "one row per claim segment",
        "CMS Inpatient Claims DE-SynPUF: each record pertains to a synthetic inpatient claim.",
    ),
    "outpatient_claims": (
        "one row per claim segment",
        "CMS Outpatient Claims DE-SynPUF: each record pertains to a synthetic outpatient claim.",
    ),
    "carrier_claims": (
        "one row per claim",
        "CMS Carrier Claims DE-SynPUF: each record pertains to a synthetic "
        "physician/supplier claim.",
    ),
}

# Owner decision (P2): this column holds diagnosis-looking codes, not procedures.
FORCED_EXCLUDES: dict[tuple[str, str], str] = {
    ("inpatient_claims", "ICD9_PRCDR_CD_6"): "excluded by owner decision",
}

_MEMBER = re.compile(r"^(.+_)(\d+)$")


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    include: bool
    distinct_count: int


@dataclass
class BuildReport:
    excluded: list[tuple[str, str, str]] = field(default_factory=list)  # table, entry, reason
    undescribed: list[tuple[str, str]] = field(default_factory=list)
    values_without_code: list[tuple[str, str, list[str]]] = field(default_factory=list)


def read_profile(path: Path) -> dict[tuple[str, str], tuple[float, int]]:
    """(table, column) -> (null_pct, distinct_count) from column_profile.csv."""
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (row["table"], row["column"]): (float(row["null_pct"]), int(row["distinct_count"]))
            for row in csv.DictReader(f)
        }


def group_columns(columns: list[ColumnInfo]) -> list[list[ColumnInfo]]:
    """Put neighbouring PREFIX_n, PREFIX_n+1, ... into one group.

    A family is split wherever type or `include` changes, so one entry never
    mixes included and excluded columns.
    """
    groups: list[list[ColumnInfo]] = []
    for column in columns:
        match = _MEMBER.match(column.name)
        if groups and match:
            prev = groups[-1][-1]
            prev_match = _MEMBER.match(prev.name)
            if (
                prev_match
                and prev_match.group(1) == match.group(1)
                and int(match.group(2)) == int(prev_match.group(2)) + 1
                and prev.type == column.type
                and prev.include == column.include
            ):
                groups[-1].append(column)
                continue
        groups.append([column])
    return groups


def entry_name(group: list[ColumnInfo]) -> str:
    if len(group) == 1:
        return group[0].name
    first, last = _MEMBER.match(group[0].name), _MEMBER.match(group[-1].name)
    return family_name(first.group(1), int(first.group(2)), int(last.group(2)))


def _is_identity(codes: dict[str, str]) -> bool:
    """True for 'code tables' that only list numbers ('00' -> '0', '1' -> '1')."""

    def same(code: str, label: str) -> bool:
        return code == label or (code.isdigit() and label.isdigit() and int(code) == int(label))

    return all(same(code, label) for code, label in codes.items())


def _values_in_data(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> set[str]:
    values: set[str] = set()
    for column in columns:
        rows = con.execute(
            f'SELECT DISTINCT CAST("{column}" AS VARCHAR) FROM {table} WHERE "{column}" IS NOT NULL'
        ).fetchall()
        values |= {r[0] for r in rows}
    return values


def build_catalogue(
    con: duckdb.DuckDBPyConnection,
    profile: dict[tuple[str, str], tuple[float, int]],
    codebook: dict[tuple[str, str], CodebookEntry],
    max_coded_distinct: int = 100,
) -> tuple[Catalogue, BuildReport]:
    report = BuildReport()
    tables = []
    for table in existing_tables(con):
        infos = []
        for name, col_type, *_ in con.execute(f"DESCRIBE {table}").fetchall():
            if (table, name) not in profile:
                raise KeyError(f"{table}.{name} is not in the column profile - re-run profile.py")
            null_pct, distinct = profile[(table, name)]
            reason = FORCED_EXCLUDES.get((table, name)) or ("100% NULL" if null_pct >= 100 else "")
            infos.append(ColumnInfo(name, col_type, include=not reason, distinct_count=distinct))

        entries = []
        for group in group_columns(infos):
            name = entry_name(group)
            members = [c.name for c in group]
            source = codebook.get((table, members[0].upper()))
            description = source.label if source else ""
            if not description:
                report.undescribed.append((table, name))
            if not group[0].include:
                reason = FORCED_EXCLUDES.get((table, members[0]), "100% NULL")
                report.excluded.append((table, name, reason))

            codes: dict[str, str] = {}
            # The distinct-count guard keeps out ID-like columns whose codebook
            # page happens to contain an unrelated "X = meaning" list.
            coded = source and source.codes and not _is_identity(source.codes)
            if coded and max(c.distinct_count for c in group) <= max_coded_distinct:
                values = _values_in_data(con, table, members)
                codes = {k: v for k, v in source.codes.items() if k in values}
                if missing := sorted(values - set(codes)):
                    report.values_without_code.append((table, name, missing))

            entries.append(
                ColumnEntry(
                    name=name,
                    type=group[0].type,
                    description=description,
                    include=group[0].include,
                    codes=codes,
                    notes="",
                )
            )
        grain, description = TABLE_INFO.get(table, ("", ""))
        tables.append(TableEntry(name=table, grain=grain, description=description, columns=entries))

    source_info = {"title": CODEBOOK_TITLE, "url": CODEBOOK_URL}
    return Catalogue(source=source_info, tables=tables), report


_HEADER = f"""\
# Schema catalogue for ClaimsSQL-Trust.
#
# Drafted by scripts/make_catalogue.py. Descriptions and code meanings are taken
# from the {CODEBOOK_TITLE}
# ({CODEBOOK_URL}).
# Reviewed by the project owner, who also writes the `notes` fields.
#
# An entry named PREFIX_1..N stands for the columns PREFIX_1 ... PREFIX_N.
# include: false keeps a column out of the prompt.
"""


def write_catalogue(catalogue: Catalogue, path: Path) -> None:
    """Write the YAML. Refuses to replace an existing file (mode 'x')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        catalogue.model_dump(), sort_keys=False, allow_unicode=True, width=110
    )
    with path.open("x", encoding="utf-8") as f:
        f.write(_HEADER + "\n" + body)
