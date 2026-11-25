"""What kind of mistake was a harmful answer? One rule-based label per case.

Checked in this order - the first that applies is the label:

  infeasible_answered  the question cannot be answered, but the system answered
  schema               the SQL uses a table or column that does not exist (V1)
  execution_error      the SQL does not run (V2)
  value                the SQL compares a low-cardinality column with a value
                       that column never takes (e.g. year = 2011, sex = 'M')
  logic                the SQL runs and returns a result, but the wrong one

The value check uses the distinct values recorded by db/profile.py
(low_cardinality_values.json). Amount columns (DOUBLE) are skipped: a filter
like "amount = 0" is not a code lookup.
"""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import traverse_scope

from claimssql.eval.score import HARMFUL, ScoredCase, verifier_outputs

INFEASIBLE_ANSWERED = "infeasible_answered"
SCHEMA = "schema"
EXECUTION_ERROR = "execution_error"
VALUE = "value"
LOGIC = "logic"
LABELS = (INFEASIBLE_ANSWERED, SCHEMA, EXECUTION_ERROR, VALUE, LOGIC)

# table (lower case) -> column (upper case) -> the values the column takes
LowCardinality = dict[str, dict[str, list]]


def normalise_low_cardinality(
    raw: dict[str, dict[str, list]], skip: set[tuple[str, str]] | None = None
) -> LowCardinality:
    """Lower/upper-case the names and drop the columns in `skip` (table, COLUMN)."""
    skip = skip or set()
    return {
        table.lower(): {
            column.upper(): values
            for column, values in columns.items()
            if (table.lower(), column.upper()) not in skip
        }
        for table, columns in raw.items()
    }


def _literal_value(node: exp.Expression) -> tuple[bool, Any] | None:
    """(is_text, value) for a literal, else None."""
    negative = isinstance(node, exp.Neg)
    if negative:
        node = node.this
    if not isinstance(node, exp.Literal):
        return None
    if node.is_string:
        return True, node.this
    try:
        number = float(node.this)
    except ValueError:
        return None
    return False, -number if negative else number


def _takes_value(values: list, is_text: bool, value: Any) -> bool:
    if is_text and all(isinstance(v, str) for v in values):
        return value in values  # text against text: exact ('1' is not '01')
    # Otherwise DuckDB compares as numbers ('01' = 1 is true).
    try:
        wanted = float(value)
    except (TypeError, ValueError):
        return value in [str(v) for v in values]
    for candidate in values:
        try:
            if float(candidate) == wanted:
                return True
        except (TypeError, ValueError):
            continue
    return False


def value_mismatches(sql: str, low_cardinality: LowCardinality) -> list[str]:
    """Comparisons (=, <>, IN) of a low-cardinality column with a literal that the
    column never takes, written like "beneficiary.YEAR = 2011"."""
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
        scopes = traverse_scope(tree) if tree is not None else []
    except SqlglotError:
        return []

    found: list[str] = []
    for scope in scopes:
        aliases = {
            alias.lower(): source.name.lower()
            for alias, (_, source) in scope.selected_sources.items()
            if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier)
        }
        tables = set(aliases.values())
        # If the query also reads from a CTE or subquery here, a bare column name
        # could belong to that, so only qualified columns are judged.
        unqualified_ok = len(aliases) == len(scope.selected_sources)

        comparisons: list[tuple[exp.Column, list[exp.Expression]]] = []
        for node in scope.find_all(exp.EQ, exp.NEQ):
            left, right = node.this, node.expression
            if isinstance(left, exp.Column):
                comparisons.append((left, [right]))
            elif isinstance(right, exp.Column):
                comparisons.append((right, [left]))
        for node in scope.find_all(exp.In):
            if isinstance(node.this, exp.Column) and node.expressions:
                comparisons.append((node.this, list(node.expressions)))

        for column, operands in comparisons:
            name = column.name.upper()
            if column.table:
                qualifier = column.table.lower()
                candidates = [aliases[qualifier]] if qualifier in aliases else []
            else:
                candidates = sorted(tables) if unqualified_ok else []
            # Judge only when exactly one table can own the column.
            owners = [t for t in candidates if name in low_cardinality.get(t, {})]
            if len(owners) != 1:
                continue
            values = low_cardinality[owners[0]][name]
            for operand in operands:
                literal = _literal_value(operand)
                if literal is not None and not _takes_value(values, *literal):
                    text = f"'{literal[1]}'" if literal[0] else f"{literal[1]:g}"
                    found.append(f"{owners[0]}.{name} = {text}")
    return found


def label_case(
    record: dict[str, Any], case: ScoredCase, low_cardinality: LowCardinality
) -> str | None:
    """The mistake label of a harmful case; None for cases that are not harmful."""
    if case.region not in HARMFUL:
        return None
    if not case.answerable:
        return INFEASIBLE_ANSWERED
    v1, v2 = verifier_outputs(record, case.config)
    if v1 and (v1.get("unknown_tables") or v1.get("unknown_columns")):
        return SCHEMA
    if not v2 or not v2.get("ok"):
        return EXECUTION_ERROR
    if case.sql and value_mismatches(case.sql, low_cardinality):
        return VALUE
    return LOGIC
