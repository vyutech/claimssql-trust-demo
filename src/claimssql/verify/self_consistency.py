"""V3 - self-consistency: do the k sampled answers agree with each other?

The idea: a model that "knows" the answer writes queries that return the same
result even when sampled at a higher temperature; a model that is guessing
scatters. So we run the k sampled queries and group them by result.

  * Samples with the same SQL (after normalisation) are executed once.
  * Samples whose results have the same hash form a cluster.
  * A sample that refused, could not be parsed, looped (DEGENERATE_OUTPUT),
    failed to execute, or returned more than the row cap agrees with nobody:
    it is its own outcome.
  * agreement = size of the largest cluster / k.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

import duckdb
import sqlglot
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

from claimssql.eval.compare import MAX_ROWS
from claimssql.verify.dry_run import DIALECT, DryRunResult, dry_run

NO_SQL = "no_sql"  # the sample refused or was a PARSE_ERROR
DEGENERATE = "degenerate"  # the model looped and the reply was aborted (DEGENERATE_OUTPUT)
FAILED = "failed"  # the SQL did not execute
TRUNCATED = "truncated"  # more rows than the cap: no reliable hash


@dataclass
class SelfConsistencyResult:
    k: int
    agreement: float  # largest cluster / k; 0.0 when no sample produced a result
    cluster_sizes: list[int]  # result clusters, largest first
    majority_hash: str | None  # hash of the largest cluster (see greedy_in_majority for ties)
    greedy_in_majority: bool
    distinct_sql_executed: int
    outcomes: list[str] = field(default_factory=list)  # per sample: a hash, or NO_SQL/FAILED/...

    def to_dict(self) -> dict:
        return asdict(self)


def normalise_sql(sql: str) -> str:
    """Canonical text of a query, so trivially different spellings compare equal:
    comments, spacing, keyword case, identifier case and a final ';' are removed."""
    try:
        tree = sqlglot.parse_one(sql, read=DIALECT)
        if tree is None:
            raise SqlglotError("empty")
        return normalize_identifiers(tree, dialect=DIALECT).sql(dialect=DIALECT, comments=False)
    except SqlglotError:
        return " ".join(sql.split()).rstrip(";").strip()  # unparseable: compare the raw text


def self_consistency(
    con: duckdb.DuckDBPyConnection,
    sample_sqls: list[str | None],
    greedy_hash: str | None,
    timeout_s: float = 30.0,
    max_rows: int = MAX_ROWS,
    sample_errors: list[str | None] | None = None,
) -> SelfConsistencyResult:
    """`sample_sqls`: the SQL of each of the k samples (None = no SQL).
    `greedy_hash`: result hash of the greedy answer, from its own dry run.
    `sample_errors`: each sample's generation error, if any; DEGENERATE_OUTPUT
    gets its own outcome so it can be counted, but like any failure it agrees
    with nobody."""
    errors = sample_errors or [None] * len(sample_sqls)
    executed: dict[str, DryRunResult] = {}
    outcomes: list[str] = []
    for sql, error in zip(sample_sqls, errors, strict=True):
        if error == "DEGENERATE_OUTPUT":
            outcomes.append(DEGENERATE)
            continue
        if not sql or not sql.strip():
            outcomes.append(NO_SQL)
            continue
        key = normalise_sql(sql)
        if key not in executed:
            executed[key] = dry_run(con, sql, timeout_s=timeout_s, max_rows=max_rows)
        result = executed[key]
        if not result.ok:
            outcomes.append(FAILED)
        elif result.result_hash is None:
            outcomes.append(TRUNCATED)
        else:
            outcomes.append(result.result_hash)

    k = len(sample_sqls)
    clusters = Counter(o for o in outcomes if o not in (NO_SQL, FAILED, TRUNCATED, DEGENERATE))
    sizes = sorted(clusters.values(), reverse=True)
    if not sizes:
        return SelfConsistencyResult(
            k=k, agreement=0.0, cluster_sizes=[], majority_hash=None, greedy_in_majority=False,
            distinct_sql_executed=len(executed), outcomes=outcomes,
        )  # fmt: skip

    largest = sizes[0]
    tied = sorted(h for h, size in clusters.items() if size == largest)
    # On a tie the greedy answer counts as "in the majority" if it is in any of
    # the tied clusters (owner decision, P5); report that cluster's hash.
    greedy_in_majority = greedy_hash is not None and greedy_hash in tied
    return SelfConsistencyResult(
        k=k,
        agreement=largest / k,
        cluster_sizes=sizes,
        majority_hash=greedy_hash if greedy_in_majority else tied[0],
        greedy_in_majority=greedy_in_majority,
        distinct_sql_executed=len(executed),
        outcomes=outcomes,
    )
