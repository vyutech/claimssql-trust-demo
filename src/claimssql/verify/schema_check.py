"""V1 - schema check: does the SQL only use tables and columns that exist?

The SQL is parsed (never executed) with sqlglot. Every table must be in the
catalogue, and every column must exist where the query says it is. Names the
query defines itself are not hallucinations: CTE names, table aliases, subquery
aliases, the output columns of CTEs and subqueries, and SELECT aliases.

The check is deliberately cautious: when it cannot know which columns a source
has (a table function, or SELECT * over an unknown table) it reports nothing
for that source. A false alarm would make the system abstain on a correct
query, which is worse for this project than missing a hallucination here -
V2 (dry run) still catches whatever fails to execute.

Matching is case-insensitive, quoted or not, as in DuckDB.

Usage (checks every gold query; they must all pass, otherwise V1 has a false positive):
    python -m claimssql.verify.schema_check
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope

from claimssql.config import REPO_ROOT
from claimssql.schema.catalogue import DEFAULT_CATALOGUE_PATH, Catalogue, load_catalogue

DIALECT = "duckdb"
Schema = dict[str, set[str]]  # table name (lower case) -> column names (upper case)


@dataclass
class SchemaCheckResult:
    unknown_tables: list[str] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)
    parse_error: str | None = None

    @property
    def passed(self) -> bool:
        return not (self.unknown_tables or self.unknown_columns or self.parse_error)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "unknown_tables": self.unknown_tables,
            "unknown_columns": self.unknown_columns,
            "parse_error": self.parse_error,
        }


def schema_from_catalogue(catalogue: Catalogue) -> Schema:
    """All catalogue columns, including include: false ones: they are real
    columns, so using one is not a hallucination (owner decision, P5)."""
    return {
        table.name.lower(): {name.upper() for name in table.column_types()}
        for table in catalogue.tables
    }


def _is_base_table(source: object) -> bool:
    """A real table reference, as opposed to a table function like range(10)."""
    return isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier)


def _owning_query(node: exp.Expression) -> exp.Expression | None:
    """The nearest enclosing SELECT or set operation of a node."""
    parent = node.parent
    while parent is not None and not isinstance(parent, (exp.Select, exp.SetOperation)):
        parent = parent.parent
    return parent


class _Checker:
    def __init__(self, schema: Schema, scopes: list[Scope]) -> None:
        self.schema = schema
        # Lets us find the scope of a set operation's branch from its expression.
        self.scope_of = {id(scope.expression): scope for scope in scopes}
        self._outputs: dict[int, set[str] | None] = {}
        self.unknown_tables: list[str] = []
        self.unknown_columns: list[str] = []

    # --- what columns does a source offer? (None = cannot know) -------------

    def source_columns(self, source: object) -> set[str] | None:
        if isinstance(source, Scope):
            # "WITH c(a, b) AS (...)" / "(...) AS s(a, b)" rename the outputs.
            wrapper = source.expression.parent
            if isinstance(wrapper, (exp.CTE, exp.Subquery)):
                alias = wrapper.args.get("alias")
                if alias is not None and alias.columns:
                    return {column.name.upper() for column in alias.columns}
            return self.output_columns(source)
        if _is_base_table(source):
            return self.schema.get(source.name.lower())  # None for an unknown table
        return None  # table function, VALUES, ...

    def output_columns(self, scope: Scope) -> set[str] | None:
        """Column names a CTE / subquery / set operation exposes to its user."""
        key = id(scope)
        if key not in self._outputs:
            self._outputs[key] = None  # guards against recursive CTEs
            self._outputs[key] = self._compute_outputs(scope)
        return self._outputs[key]

    def _compute_outputs(self, scope: Scope) -> set[str] | None:
        expression = scope.expression
        while isinstance(expression, exp.SetOperation):
            # A set operation takes its column names from its first branch.
            expression = expression.left.unnest()
            branch = self.scope_of.get(id(expression))
            if branch is None:
                return None
            scope = branch
        if not isinstance(expression, exp.Select):
            return None  # UNNEST, VALUES, ...

        names: set[str] = set()
        for projection in expression.selects:
            star = projection if isinstance(projection, exp.Star) else None
            qualifier = None
            if isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
                star, qualifier = projection.this, projection.table  # t.*
            if star is None:
                if name := projection.alias_or_name:
                    names.add(name.upper())
                continue
            sources = scope.selected_sources
            picked = (
                [sources[qualifier][1]]
                if qualifier and qualifier in sources
                else [source for _, source in sources.values()]
            )
            for source in picked:
                columns = self.source_columns(source)
                if columns is None:
                    return None  # * over something we cannot see into
                names |= columns
            # (* EXCLUDE (...) would only remove names; keeping them is the safe side.)
        return names

    # --- the checks ----------------------------------------------------------

    def check(self, scope: Scope) -> None:
        for _, source in scope.selected_sources.values():
            if _is_base_table(source) and source.name.lower() not in self.schema:
                self._add(self.unknown_tables, source.name)
        if scope.pivots:
            return  # PIVOT / UNPIVOT invent column names of their own
        for column in scope.columns:
            if isinstance(column.this, exp.Star):
                continue
            if _owning_query(column) is not scope.expression:
                # sqlglot also lists the columns of nested scalar subqueries
                # here; each of those is checked in its own scope.
                continue
            if column.table:
                self._check_qualified(scope, column)
            else:
                self._check_unqualified(scope, column)

    def _find_source(self, scope: Scope | None, alias: str) -> tuple[bool, object]:
        """Look for a table alias here, then in enclosing queries (correlation)."""
        wanted = alias.lower()
        while scope is not None:
            for name, (_, source) in scope.selected_sources.items():
                if name.lower() == wanted:
                    return True, source
            scope = scope.parent
        return False, None

    def _check_qualified(self, scope: Scope, column: exp.Column) -> None:
        found, source = self._find_source(scope, column.table)
        if found:
            columns = self.source_columns(source)
            label = source.name if _is_base_table(source) else column.table
        elif column.table.lower() in self.schema:
            # e.g. "beneficiary.X" written although the table was given an alias.
            columns, label = self.schema[column.table.lower()], column.table
        else:
            return  # undefined alias or struct field: not a schema question, V2 decides
        if columns is not None and column.name.upper() not in columns:
            self._add(self.unknown_columns, f"{label}.{column.name}")

    def _check_unqualified(self, scope: Scope, column: exp.Column) -> None:
        name = column.name.upper()
        current: Scope | None = scope
        while current is not None:  # this query first, then enclosing queries
            expression = current.expression
            if isinstance(expression, exp.SetOperation):
                # ORDER BY after a UNION refers to the UNION's output columns.
                outputs = self.output_columns(current)
                if outputs is None or name in outputs:
                    return
            else:
                for _, source in current.selected_sources.values():
                    columns = self.source_columns(source)
                    if columns is None or name in columns:
                        return  # found, or a source we cannot see into
                if isinstance(expression, exp.Select) and any(
                    p.alias and p.alias.upper() == name for p in expression.selects
                ):
                    return  # a SELECT alias reused in WHERE / GROUP BY / ORDER BY / QUALIFY
            current = current.parent
        self._add(self.unknown_columns, column.name)

    @staticmethod
    def _add(target: list[str], name: str) -> None:
        if name.lower() not in {existing.lower() for existing in target}:
            target.append(name)


def check_sql(sql: str, schema: Schema) -> SchemaCheckResult:
    try:
        statements = sqlglot.parse(sql, read=DIALECT)
    except SqlglotError as exc:
        return SchemaCheckResult(parse_error=str(exc).splitlines()[0])
    statements = [s for s in statements if s is not None]
    if not statements:
        return SchemaCheckResult(parse_error="empty SQL")

    result = SchemaCheckResult()
    for tree in statements:
        try:
            scopes = traverse_scope(tree)
        except SqlglotError as exc:
            return SchemaCheckResult(parse_error=f"cannot analyse query: {exc}")
        checker = _Checker(schema, scopes)
        if not scopes:
            # Not a query (e.g. DROP TABLE x): still report unknown table names.
            for table in tree.find_all(exp.Table):
                if _is_base_table(table) and table.name.lower() not in schema:
                    checker._add(checker.unknown_tables, table.name)
        for scope in scopes:
            checker.check(scope)
        for name in checker.unknown_tables:
            _Checker._add(result.unknown_tables, name)
        for name in checker.unknown_columns:
            _Checker._add(result.unknown_columns, name)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V1 on every gold query.")
    parser.add_argument("--questions", type=Path,
                        default=REPO_ROOT / "benchmark" / "questions.jsonl")
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE_PATH)
    args = parser.parse_args(argv)

    schema = schema_from_catalogue(load_catalogue(args.catalogue))
    checked = failed = 0
    for line in args.questions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("gold_sql"):
            continue
        checked += 1
        result = check_sql(record["gold_sql"], schema)
        if not result.passed:
            failed += 1
            print(f"{record['id']}: tables={result.unknown_tables} "
                  f"columns={result.unknown_columns} parse_error={result.parse_error}")
    print(f"V1 on gold SQL: {checked} checked, {checked - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
